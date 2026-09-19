"""Lazy, server-side access to the Google Earth Engine Sentinel-2 catalogue.

The provider deliberately does not authenticate until a search or preview is
requested.  That keeps the public configuration endpoint free of network calls
and prevents credentials from ever reaching a browser response.
"""

from __future__ import annotations

import importlib
import math
import re
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable
from urllib.parse import parse_qsl, urlsplit


COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
SCENE_ID = re.compile(
    r"^COPERNICUS/S2_SR_HARMONIZED/"
    r"\d{8}T\d{6}_\d{8}T\d{6}_T\d{2}[A-Z]{3}$"
)
SCOPES = (
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
)
TILE_HOSTS = {"earthengine.googleapis.com", "earthengine-highvolume.googleapis.com"}
# ee._state and its Cloud HTTP resources are process-wide. The API gateway may
# execute synchronous provider calls on more than one worker thread, so every
# call that touches the SDK (including initialization) shares this zero-queue
# lock. A busy caller receives 429 instead of running after its HTTP deadline.
_EE_CALL_LOCK = threading.RLock()


class ImageryError(Exception):
    """A deliberately safe error which the HTTP layer can expose."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class EarthEngineProvider:
    """Bounded synchronous Earth Engine queries for Sentinel-2 visual search."""

    def __init__(
        self,
        project: str | None,
        enabled: bool = False,
        timeout_seconds: int = 20,
    ) -> None:
        self.project = project.strip() if isinstance(project, str) else None
        self.enabled = enabled
        self.timeout_seconds = max(1, int(timeout_seconds))
        self._initialized = False
        self._lock = threading.Lock()

    def configuration(self) -> dict[str, Any]:
        """Return local prerequisites only; do not import or contact Google."""
        result: dict[str, Any] = {
            "provider": "earth_engine",
            "configured": bool(self.enabled and self.project),
            "collection": COLLECTION,
        }
        if not self.enabled:
            result["reason"] = "disabled"
        elif not self.project:
            result["reason"] = "project_not_configured"
        return result

    def _dependencies(self) -> tuple[Any, Callable[..., Any]]:
        try:
            ee = importlib.import_module("ee")
            google_auth = importlib.import_module("google.auth")
            return ee, google_auth.default
        except (ImportError, AttributeError) as exc:
            raise ImageryError(
                503, "GOOGLE_NOT_CONFIGURED", "Google Earth Engine is not configured"
            ) from exc

    def _ensure_initialized(self) -> Any:
        config = self.configuration()
        if not config["configured"]:
            raise ImageryError(
                503,
                "GOOGLE_NOT_CONFIGURED",
                "Google Earth Engine is not configured",
            )
        with self._lock:
            if self._initialized:
                return self._dependencies()[0]
            ee, credentials_default = self._dependencies()
            try:
                data = getattr(ee, "data", None)
                if data is not None:
                    set_retries = getattr(data, "setMaxRetries", None)
                    if callable(set_retries):
                        set_retries(0)
            except Exception as exc:
                raise ImageryError(
                    503,
                    "EARTH_ENGINE_UNAVAILABLE",
                    "Google Earth Engine is temporarily unavailable",
                ) from exc
            try:
                # ADC also honors GOOGLE_APPLICATION_CREDENTIALS when it points at
                # a service-account JSON file.  Credentials remain process-local.
                credentials, _ = credentials_default(scopes=list(SCOPES))
            except Exception as exc:
                raise ImageryError(
                    503,
                    "GOOGLE_NOT_CONFIGURED",
                    "Google Earth Engine credentials are not configured",
                ) from exc
            try:
                # In EE 1.7.43 setDeadline() rebuilds the Cloud resource and
                # requires a session created by Initialize(). A bounded custom
                # transport covers discovery and credential refresh as well as
                # subsequent calls, including the cold-start initialization.
                httplib2 = importlib.import_module("httplib2")
                ee.Initialize(
                    credentials=credentials,
                    project=self.project,
                    http_transport=httplib2.Http(timeout=self.timeout_seconds),
                )
                if data is not None:
                    set_deadline = getattr(data, "setDeadline", None)
                    if callable(set_deadline):
                        set_deadline(self.timeout_seconds * 1000)
                self._initialized = True
                return ee
            except Exception as exc:
                raise ImageryError(
                    503,
                    "EARTH_ENGINE_UNAVAILABLE",
                    "Google Earth Engine is temporarily unavailable",
                ) from exc

    @staticmethod
    def _geometry(ee: Any, aoi: dict[str, Any]) -> Any:
        try:
            return ee.Geometry(aoi)
        except Exception as exc:
            raise ImageryError(
                422, "INVALID_AOI", "AOI cannot be read by Earth Engine"
            ) from exc

    @staticmethod
    def _scene_id(feature: dict[str, Any]) -> str:
        props = (
            feature.get("properties")
            if isinstance(feature.get("properties"), dict)
            else {}
        )
        scene_id = (
            props.get("id")
            or props.get("system:id")
            or feature.get("id")
            or props.get("PRODUCT_ID")
        )
        if not isinstance(scene_id, str):
            return ""
        return (
            scene_id
            if scene_id.startswith(COLLECTION + "/")
            else f"{COLLECTION}/{scene_id}"
        )

    @staticmethod
    def _bounds(geometry: Any) -> list[float]:
        points: list[tuple[float, float]] = []

        def walk(value: Any) -> None:
            if isinstance(value, (list, tuple)):
                if len(value) >= 2 and all(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and math.isfinite(float(item))
                    for item in value[:2]
                ):
                    points.append((float(value[0]), float(value[1])))
                else:
                    for item in value:
                        walk(item)

        if isinstance(geometry, dict):
            walk(geometry.get("coordinates"))
        if not points:
            raise ImageryError(
                503,
                "EARTH_ENGINE_UNAVAILABLE",
                "Earth Engine returned invalid scene geometry",
            )
        longitudes, latitudes = zip(*points)
        return [min(longitudes), min(latitudes), max(longitudes), max(latitudes)]

    @staticmethod
    def _acquired_at(value: Any) -> str:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ImageryError(
                503,
                "EARTH_ENGINE_UNAVAILABLE",
                "Earth Engine returned invalid scene time",
            )
        return (
            datetime.fromtimestamp(value / 1000, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _search_inputs(payload: dict[str, Any]) -> tuple[date, date, int, float]:
        try:
            start = date.fromisoformat(payload["date_start"])
            end = date.fromisoformat(payload["date_end"])
            limit = int(payload["limit"])
            cloud_max = float(payload["cloud_max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ImageryError(
                422, "INVALID_SEARCH", "search input is invalid"
            ) from exc
        if end < start:
            raise ImageryError(
                422, "INVALID_DATE_RANGE", "date_end precedes date_start"
            )
        if limit < 1 or not math.isfinite(cloud_max):
            raise ImageryError(422, "INVALID_SEARCH", "search input is invalid")
        return start, end + timedelta(days=1), limit, cloud_max

    @classmethod
    def _local_aoi_bounds(cls, aoi: Any) -> list[float]:
        try:
            return cls._bounds(aoi)
        except ImageryError as exc:
            raise ImageryError(422, "INVALID_AOI", "AOI is invalid") from exc

    @staticmethod
    def _tile_url(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Earth Engine did not return a tile URL")
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in TILE_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or any(key.lower() == "access_token" for key, _ in parse_qsl(parsed.query))
        ):
            raise ValueError("Earth Engine returned an unsafe tile URL")
        return value

    def _search(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return at most ``limit`` newest scene summaries from one bounded call."""
        start, end_exclusive, limit, cloud_max = self._search_inputs(payload)
        aoi_payload = payload.get("aoi")
        self._local_aoi_bounds(aoi_payload)
        ee = self._ensure_initialized()
        try:
            aoi = self._geometry(ee, aoi_payload)
            collection = (
                ee.ImageCollection(COLLECTION)
                .filterBounds(aoi)
                .filterDate(start.isoformat(), end_exclusive.isoformat())
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
                .sort("system:time_start", False)
                .limit(limit + 1)
            )
            images = collection.toList(limit + 1)
            features = images.map(
                lambda image: ee.Feature(
                    ee.Image(image).geometry().bounds(),
                    {
                        "id": ee.Image(image).id(),
                        "system:time_start": ee.Image(image).get("system:time_start"),
                        "CLOUDY_PIXEL_PERCENTAGE": ee.Image(image).get(
                            "CLOUDY_PIXEL_PERCENTAGE"
                        ),
                    },
                )
            )
            raw = ee.FeatureCollection(features).getInfo()
            if not isinstance(raw, dict):
                raise ValueError("Earth Engine returned malformed scene data")
            features = raw.get("features")
            if not isinstance(features, list):
                raise ValueError("features are not a list")
            scenes: list[dict[str, Any]] = []
            for feature in features[:limit]:
                if not isinstance(feature, dict):
                    raise ValueError("feature is not an object")
                props = (
                    feature.get("properties")
                    if isinstance(feature.get("properties"), dict)
                    else {}
                )
                cloud = props.get("CLOUDY_PIXEL_PERCENTAGE")
                scene_id = self._scene_id(feature)
                if not SCENE_ID.fullmatch(scene_id):
                    raise ValueError("Earth Engine returned an invalid scene id")
                if cloud is not None and (
                    not isinstance(cloud, (int, float))
                    or isinstance(cloud, bool)
                    or not math.isfinite(float(cloud))
                ):
                    raise ValueError("Earth Engine returned invalid cloud cover")
                scenes.append(
                    {
                        "id": scene_id,
                        "acquired_at": self._acquired_at(
                            props.get("system:time_start")
                        ),
                        "cloud_percent": float(cloud) if cloud is not None else None,
                        "bounds": self._bounds(feature.get("geometry")),
                        "sensor": "Sentinel-2",
                        "resolution_m": 10,
                    }
                )
            return {
                "provider": "earth_engine",
                "collection": COLLECTION,
                "scenes": scenes,
                "truncated": len(features) > limit,
            }
        except ImageryError:
            raise
        except Exception as exc:
            raise ImageryError(
                503,
                "EARTH_ENGINE_UNAVAILABLE",
                "Google Earth Engine is temporarily unavailable",
            ) from exc

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one full Earth Engine search or reject when the SDK is busy."""
        if not _EE_CALL_LOCK.acquire(blocking=False):
            raise ImageryError(
                429, "IMAGERY_BUSY", "Earth Engine request capacity reached"
            )
        try:
            return self._search(payload)
        finally:
            _EE_CALL_LOCK.release()

    def _preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build a true-colour Earth Engine tile URL for an allowlisted scene."""
        scene_id = payload.get("scene_id")
        if not isinstance(scene_id, str) or not SCENE_ID.fullmatch(scene_id):
            raise ImageryError(
                422, "INVALID_SCENE_ID", "scene_id is not a Sentinel-2 scene"
            )
        aoi_payload = payload.get("aoi")
        aoi_bounds = self._local_aoi_bounds(aoi_payload)
        ee = self._ensure_initialized()
        try:
            aoi = self._geometry(ee, aoi_payload)
            image = (
                ee.Image(scene_id)
                .select(["B4", "B3", "B2"])
                .clip(aoi)
                .visualize(min=0, max=3000)
            )
            map_id = image.getMapId({})
            tile_url = self._tile_url(map_id["tile_fetcher"].url_format)
            return {
                "scene_id": scene_id,
                "tile_url": tile_url,
                "attribution": "Contains modified Copernicus Sentinel data, processed by Google Earth Engine",
                "bounds": aoi_bounds,
            }
        except ImageryError:
            raise
        except Exception as exc:
            raise ImageryError(
                503,
                "EARTH_ENGINE_UNAVAILABLE",
                "Google Earth Engine is temporarily unavailable",
            ) from exc

    def preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one full Earth Engine preview or reject when the SDK is busy."""
        if not _EE_CALL_LOCK.acquire(blocking=False):
            raise ImageryError(
                429, "IMAGERY_BUSY", "Earth Engine request capacity reached"
            )
        try:
            return self._preview(payload)
        finally:
            _EE_CALL_LOCK.release()
