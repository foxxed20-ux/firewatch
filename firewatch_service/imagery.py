"""Bounded, independent scene discovery; it never registers model input data."""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date
import hashlib
import json
import math
import threading
import time
from typing import Any, Protocol

from pyproj import Transformer
from shapely.geometry import shape

from .config import Settings
from .imagery_provider import EarthEngineProvider, ImageryError

MAX_WINDOW_EDGE_METERS = 300_000


class ImageryProvider(Protocol):
    def configuration(self) -> dict[str, Any]: ...
    def search(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def preview(self, payload: dict[str, Any]) -> dict[str, Any]: ...


def validate_limits(payload: dict[str, Any], settings: Settings) -> None:
    geometry = shape(payload["aoi"])
    west, south, east, north = geometry.bounds
    if east - west > 180 or south < -85 or north > 85:
        raise ImageryError(
            422, "IMAGERY_AOI_INVALID", "AOI must avoid the antimeridian and polar caps"
        )
    # Bound the full window, including gaps/holes. A thin or disjoint polygon
    # must not turn a cheap area check into a continent-wide EE search/preview.
    project = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    x1, y1 = project.transform(west, south)
    x2, y2 = project.transform(east, north)
    width, height = abs(x2 - x1), abs(y2 - y1)
    area_km2 = width * height / 1_000_000
    if (
        not math.isfinite(area_km2)
        or area_km2 > settings.imagery_max_area_km2
        or max(width, height) > MAX_WINDOW_EDGE_METERS
    ):
        raise ImageryError(
            422,
            "IMAGERY_AOI_TOO_LARGE",
            "AOI bounding area exceeds limit or its width/height exceeds 300 km",
        )
    coordinates = payload["aoi"]["coordinates"]
    polygons = [coordinates] if payload["aoi"]["type"] == "Polygon" else coordinates
    positions = [point for polygon in polygons for ring in polygon for point in ring]
    if len(positions) > 1000 or any(len(point) != 2 for point in positions):
        raise ImageryError(
            422,
            "IMAGERY_AOI_INVALID",
            "AOI supports at most 1000 longitude/latitude pairs",
        )
    if "date_start" in payload:
        days = (
            date.fromisoformat(payload["date_end"])
            - date.fromisoformat(payload["date_start"])
        ).days + 1
        if days > settings.imagery_max_date_days:
            raise ImageryError(
                422, "IMAGERY_DATE_RANGE_TOO_LARGE", "Date range exceeds limit"
            )


class ImageryGateway:
    """Caps cost and concurrency per API process, even after caller timeouts.

    Running SDK threads cannot safely be killed. They keep their capacity slot
    until the actual call ends; subsequent requests fail fast instead of growing
    an unbounded executor queue. The SDK also has its own HTTP deadline.
    """

    def __init__(self, settings: Settings, provider: ImageryProvider | None = None):
        self.settings = settings
        self.provider = provider or EarthEngineProvider(
            project=settings.google_cloud_project,
            enabled=settings.earth_engine_enabled,
            timeout_seconds=settings.imagery_timeout_seconds,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=settings.imagery_max_concurrent,
            thread_name_prefix="firewatch-imagery",
        )
        self._capacity = threading.BoundedSemaphore(settings.imagery_max_concurrent)
        self._lock = threading.Lock()
        self._starts: deque[float] = deque()
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def client_config(self) -> dict[str, Any]:
        key = self.settings.google_maps_api_key
        return {
            "maps": {
                "provider": "google" if key else "osm",
                "api_key": key,
                "configured": bool(key),
            },
            "imagery": self.provider.configuration(),
        }

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def execute(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        validate_limits(payload, self.settings)
        if not self.provider.configuration()["configured"]:
            raise ImageryError(
                503, "GOOGLE_NOT_CONFIGURED", "Earth Engine is not configured"
            )
        key = hashlib.sha256(
            json.dumps(
                [operation, payload], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                if cached[0] > now:
                    self._cache.move_to_end(key)
                    return deepcopy(cached[1])
                del self._cache[key]
            while self._starts and self._starts[0] <= now - 60:
                self._starts.popleft()
            if len(self._starts) >= self.settings.imagery_rate_limit_per_minute:
                raise ImageryError(
                    429, "IMAGERY_RATE_LIMITED", "Earth Engine request budget reached"
                )
            if not self._capacity.acquire(blocking=False):
                raise ImageryError(
                    429, "IMAGERY_BUSY", "Earth Engine request capacity reached"
                )
            self._starts.append(now)
        try:
            function = (
                self.provider.search if operation == "search" else self.provider.preview
            )
            future = self._executor.submit(function, deepcopy(payload))
        except Exception:
            self._capacity.release()
            raise ImageryError(
                503, "EARTH_ENGINE_UNAVAILABLE", "Earth Engine is unavailable"
            ) from None
        future.add_done_callback(lambda _: self._capacity.release())
        try:
            result = await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=self.settings.imagery_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise ImageryError(
                504, "EARTH_ENGINE_TIMEOUT", "Earth Engine request timed out"
            ) from None
        except ImageryError:
            raise
        except Exception:
            # Provider exception strings can include credentials or signed URLs.
            raise ImageryError(
                503, "EARTH_ENGINE_UNAVAILABLE", "Earth Engine is unavailable"
            ) from None
        with self._lock:
            self._cache[key] = (
                time.monotonic() + self.settings.imagery_cache_seconds,
                deepcopy(result),
            )
            self._cache.move_to_end(key)
            while len(self._cache) > self.settings.imagery_cache_entries:
                self._cache.popitem(last=False)
        return result
