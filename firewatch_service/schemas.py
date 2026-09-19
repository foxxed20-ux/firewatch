from __future__ import annotations

from datetime import date, datetime, timezone
from math import isfinite
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Task = Literal["af", "bs"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BBoxAOI(StrictModel):
    bbox: Annotated[list[float], Field(min_length=4, max_length=4)]

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, values: list[float]) -> list[float]:
        if not all(isfinite(v) for v in values):
            raise ValueError("bbox coordinates must be finite")
        west, south, east, north = values
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError(
                "bbox must be [west, south, east, north] in WGS84 without antimeridian crossing"
            )
        return values


class GeometryAOI(StrictModel):
    geometry: dict[str, Any]

    @field_validator("geometry")
    @classmethod
    def validate_geometry(cls, geometry: dict[str, Any]) -> dict[str, Any]:
        kind = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if (
            kind not in {"Polygon", "MultiPolygon"}
            or not isinstance(coordinates, list)
            or not coordinates
        ):
            raise ValueError(
                "geometry must be a non-empty GeoJSON Polygon or MultiPolygon"
            )
        vertices = 0

        def ring_ok(ring: Any) -> bool:
            nonlocal vertices
            if not isinstance(ring, list) or len(ring) < 4 or ring[0] != ring[-1]:
                return False
            vertices += len(ring)
            for position in ring:
                if not isinstance(position, list) or len(position) < 2:
                    return False
                lon, lat = position[0], position[1]
                if (
                    not isinstance(lon, (int, float))
                    or not isinstance(lat, (int, float))
                    or not isfinite(lon)
                    or not isfinite(lat)
                    or not -180 <= lon <= 180
                    or not -90 <= lat <= 90
                ):
                    return False
            return True

        polygons = [coordinates] if kind == "Polygon" else coordinates
        if not all(
            isinstance(polygon, list)
            and polygon
            and all(ring_ok(ring) for ring in polygon)
            for polygon in polygons
        ):
            raise ValueError("geometry rings must be closed WGS84 coordinate arrays")
        if vertices > 10_000:
            raise ValueError("geometry has too many vertices")
        try:
            from shapely.geometry import shape

            candidate = shape(geometry)
            if candidate.is_empty or not candidate.is_valid or candidate.area <= 0:
                raise ValueError("geometry must be valid and have non-zero area")
        except ImportError:
            # The production environment includes Shapely via the geo runtime.
            # Retain structural validation for the narrow API-only installation.
            pass
        return geometry


class Period(StrictModel):
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("RFC3339 timestamp must include timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def ordered(self) -> "Period":
        if self.start >= self.end:
            raise ValueError("period start must precede end")
        return self


class AnalysisRequest(StrictModel):
    dataset_id: Annotated[
        str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    ]
    model_bundle_id: Annotated[
        str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    ]
    aoi: BBoxAOI | GeometryAOI
    period: Period
    tasks: Annotated[list[Task], Field(min_length=1, max_length=2)]

    @field_validator("tasks")
    @classmethod
    def unique_tasks(cls, values: list[Task]) -> list[Task]:
        if len(values) != len(set(values)):
            raise ValueError("tasks must be unique")
        return values


class PredictionRequest(StrictModel):
    dataset_id: Annotated[
        str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    ]
    model_bundle_id: Annotated[
        str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    ]
    chip_ids: Annotated[list[str], Field(min_length=1, max_length=128)]

    @field_validator("chip_ids")
    @classmethod
    def valid_chips(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)) or any(
            not value or len(value) > 256 or "/" in value or "\\" in value
            for value in values
        ):
            raise ValueError("chip_ids must be unique safe catalog identifiers")
        return values


class APIError(StrictModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorEnvelope(StrictModel):
    error: APIError


class ImageryPreviewRequest(StrictModel):
    scene_id: Annotated[str, Field(min_length=1, max_length=160)]
    aoi: dict[str, Any]

    @field_validator("aoi")
    @classmethod
    def valid_imagery_aoi(cls, value: dict[str, Any]) -> dict[str, Any]:
        return GeometryAOI(geometry=value).geometry


class ImagerySearchRequest(StrictModel):
    aoi: dict[str, Any]
    date_start: date
    date_end: date
    cloud_max: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)] = 30
    limit: Annotated[int, Field(ge=1, le=20, strict=True)] = 10

    @field_validator("aoi")
    @classmethod
    def valid_imagery_aoi(cls, value: dict[str, Any]) -> dict[str, Any]:
        return GeometryAOI(geometry=value).geometry

    @field_validator("date_start", "date_end", mode="before")
    @classmethod
    def calendar_date_only(cls, value: Any) -> Any:
        # Do not accept epoch timestamps, datetimes, or implicit timezone shifts.
        if not isinstance(value, str) or len(value) != 10:
            raise ValueError("imagery dates must use YYYY-MM-DD")
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            raise ValueError("imagery dates must use YYYY-MM-DD") from None
        if parsed.isoformat() != value or parsed == date.max:
            raise ValueError("imagery dates must use YYYY-MM-DD before 9999-12-31")
        return value

    @model_validator(mode="after")
    def ordered_dates(self) -> "ImagerySearchRequest":
        if self.date_start > self.date_end:
            raise ValueError("date_start must not follow date_end")
        return self
