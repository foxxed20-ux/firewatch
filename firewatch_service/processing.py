"""Model orchestration and geospatial exports for prepared demo scenes."""

from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import describe_models, discover_datasets


def _utc(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _request_aoi(request: dict[str, Any]):
    from shapely.geometry import box, shape

    aoi = request.get("aoi", {})
    if isinstance(aoi, dict) and isinstance(aoi.get("geometry"), dict):
        aoi = aoi["geometry"]
    if (
        isinstance(aoi, dict)
        and isinstance(aoi.get("bbox"), list)
        and len(aoi["bbox"]) == 4
    ):
        return box(*map(float, aoi["bbox"]))
    if isinstance(aoi, dict) and aoi.get("type"):
        return shape(aoi)
    raise ValueError("aoi must be GeoJSON or {bbox:[west,south,east,north]}")


def _scene_geometry(scene: dict[str, Any], base: Path):
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import transform_bounds

    raster = scene.get("reference_raster")
    if raster:
        import rasterio

        with rasterio.open(base / raster) as ds:
            bounds = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
        from shapely.geometry import box

        return box(*bounds)
    if all(k in scene for k in ("crs", "transform", "width", "height")):
        transform = Affine(*scene["transform"])
        width, height = int(scene["width"]), int(scene["height"])
        from rasterio.warp import transform_bounds
        from shapely.geometry import box

        b = transform_bounds(
            CRS.from_user_input(scene["crs"]),
            "EPSG:4326",
            *__import__("rasterio").transform.array_bounds(height, width, transform),
            densify_pts=21,
        )
        return box(*b)
    raise ValueError(f"scene {scene.get('scene_id')} has no georeference")


def _choose_dataset(data_root: Path, dataset_id: str) -> dict[str, Any]:
    for item in discover_datasets(data_root):
        if item["dataset_id"] == dataset_id:
            return item
    raise ValueError(f"unknown demo dataset: {dataset_id}")


def snapshot_request(
    kind: str, request: dict[str, Any], data_root: Path, model_root: Path
) -> dict[str, Any]:
    """Pin catalog metadata and immutable release model path at acceptance.

    This private payload is persisted separately from the public idempotency hash.
    Dataset raster directories are immutable by the operator's publication policy.
    """
    manifest = _choose_dataset(data_root, request["dataset_id"])
    selected = (
        _selected_scenes(request, manifest)
        if kind == "analysis"
        else [s for s in manifest["scenes"] if s.get("chip_id") in request["chip_ids"]]
    )
    document = {k: v for k, v in manifest.items() if k != "_path"}
    revision = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    frozen = {**document, "_path": str(manifest["_path"].resolve()), "scenes": selected}
    model_path = model_root.resolve() / "manifest.json"
    return {
        **request,
        "_catalog_snapshot": frozen,
        "_catalog_revision": revision,
        "_model_root": str(model_root.resolve()),
        "_model_manifest_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()
        if model_path.is_file()
        else None,
    }


def _manifest_for_request(
    request: dict[str, Any], data_root: Path, model_root: Path
) -> dict[str, Any]:
    expected = request.get("_model_manifest_sha256")
    if (
        expected
        and hashlib.sha256((model_root / "manifest.json").read_bytes()).hexdigest()
        != expected
    ):
        raise ValueError("Model manifest changed after job acceptance")
    frozen = request.get("_catalog_snapshot")
    if isinstance(frozen, dict):
        return {**frozen, "_path": Path(frozen["_path"])}
    return _choose_dataset(data_root, str(request.get("dataset_id", "")))


def _selected_scenes(
    request: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    aoi = _request_aoi(request)
    period = request.get("period", {})
    start, end = _utc(period.get("start")), _utc(period.get("end"))
    requested = set(request.get("tasks", ["af", "bs"]))
    base = manifest["_path"].parent
    selected = []
    for scene in manifest.get("scenes", []):
        if not isinstance(scene, dict) or scene.get("task") not in requested:
            continue
        if request.get("chip_id") and str(scene.get("chip_id")) != str(
            request["chip_id"]
        ):
            continue
        if scene.get("task") == "af":
            observed = _utc(scene.get("observed_at"))
            if (
                observed is None
                or (start and observed < start)
                or (end and observed >= end)
            ):
                continue
        else:
            pre, post = _utc(scene.get("date_pre")), _utc(scene.get("date_post"))
            if (
                pre is None
                or post is None
                or pre >= post
                or (start and post < start)
                or (end and post >= end)
            ):
                continue
        if _scene_geometry(scene, base).intersects(aoi):
            selected.append(scene)
    return selected


def _model_ready(
    model_root: Path, bundle_id: str | None
) -> tuple[bool, dict[str, Any] | None]:
    payload = describe_models(model_root)
    for model in payload["models"]:
        if (bundle_id is None or model["model_bundle_id"] == bundle_id) and model[
            "status"
        ] == "ready":
            return True, model
    return False, None


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )


def _no_data_result(
    request: dict[str, Any],
    aoi,
    output: Path,
    reason: str,
    counts: dict[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    requested = list(dict.fromkeys(request.get("tasks", [])))
    area = _area_ha(__import__("shapely").geometry.mapping(aoi), "EPSG:4326")
    counts = counts or {}
    modules = {
        task: {
            "data_status": "no_data",
            **(
                {"hotspot_count": None}
                if task == "af"
                else {"burn_area_ha": None, "severity_area_ha": None}
            ),
        }
        for task in requested
    }
    quality = {
        task: {
            "observed_area_ha": 0.0,
            "unobserved_area_ha": area,
            "coverage_fraction": 0.0,
            "cloud_fraction": None,
            "observation_count": counts.get(task, 0),
        }
        for task in requested
    }
    result = {"modules": modules, "quality": {**quality, "warnings": [reason]}}
    _write_json(output / "summary.json", result)
    _write_json(output / "provenance.json", {"reason": reason})
    return result, {"summary": "summary.json", "provenance": "provenance.json"}


def _transform_for(scene: dict[str, Any], base: Path):
    import rasterio
    from rasterio.transform import Affine

    raster = scene.get("reference_raster")
    if raster:
        with rasterio.open(base / raster) as ds:
            return ds.transform, ds.crs
    return Affine(*scene["transform"]), scene["crs"]


def _to_wgs84(geometry: dict, crs: Any) -> dict:
    from rasterio.warp import transform_geom

    return transform_geom(crs, "EPSG:4326", geometry, precision=7)


@lru_cache(maxsize=32)
def _area_transformer(crs: str):
    from pyproj import Transformer

    return Transformer.from_crs(crs, "EPSG:6933", always_xy=True).transform


def _area_ha(geometry: dict, crs: Any) -> float:
    from shapely.geometry import shape
    from shapely.ops import transform

    return transform(_area_transformer(str(crs)), shape(geometry)).area / 10_000.0


def _polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        if not geometry.is_empty and geometry.area > 0:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _polygon_parts(part)


def _prediction_exports(
    predictions: list[tuple[dict[str, Any], dict[str, Any]]],
    manifest: dict[str, Any],
    aoi,
    output: Path,
    bundle_id: str,
    requested_tasks: list[str] | None = None,
) -> dict[str, Any]:
    """Export clipped products; BS pixels are mosaicked newest-post first."""
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import xy
    from shapely.geometry import Point, mapping, shape
    from shapely.ops import unary_union

    base = manifest["_path"].parent
    aoi_area = _area_ha(mapping(aoi), "EPSG:4326")
    hotspots, contours, provenance = [], [], []
    af_cells = {}
    unrounded_areas = {str(k): [] for k in (1, 2, 3)}
    pixel_rles: dict[str, dict[str, str]] = {}
    coverage: dict[str, list[Any]] = {"af": [], "bs": []}
    bs_records = []
    masks: dict[str, str] = {}
    for scene, result in predictions:
        raw_cmap, valid = (
            np.asarray(result["class_map"]),
            np.asarray(result["observation_valid"], dtype=bool),
        )
        allowed = [0, 1] if scene["task"] == "af" else [0, 1, 2, 3]
        if (
            raw_cmap.ndim != 2
            or raw_cmap.shape != valid.shape
            or not np.issubdtype(raw_cmap.dtype, np.number)
            or not np.isin(raw_cmap, allowed).all()
        ):
            raise ValueError(
                f"invalid class_map/valid_mask from {scene.get('chip_id')}"
            )
        cmap = raw_cmap.astype(np.uint8)
        transform_affine, crs = _transform_for(scene, base)
        reference = scene.get("reference_raster")
        if reference:
            with rasterio.open(base / reference) as ds:
                if (ds.height, ds.width) != cmap.shape:
                    raise ValueError(
                        f"prediction dimensions differ from reference raster for {scene['scene_id']}"
                    )
        filename = f"mask_{scene['scene_id']}.tif"
        with rasterio.open(
            output / filename,
            "w",
            driver="GTiff",
            height=cmap.shape[0],
            width=cmap.shape[1],
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform_affine,
            nodata=255,
            compress="deflate",
        ) as dst:
            dst.write(np.where(valid, cmap, 255).astype("uint8"), 1)
        masks[str(scene["scene_id"])] = filename
        from competition.rle import encode_bs_rles, encode_rle

        competition_map = np.where(valid, cmap, 0).astype("uint8")
        pixel_rles[str(scene["chip_id"])] = (
            {"1": encode_rle(competition_map == 1)}
            if scene["task"] == "af"
            else {str(k): v for k, v in encode_bs_rles(competition_map).items()}
        )
        valid_geoms = [
            shape(_to_wgs84(g, crs)).intersection(aoi)
            for g, _ in shapes(
                valid.astype("uint8"),
                mask=valid,
                transform=transform_affine,
                connectivity=4,
            )
        ]
        coverage[scene["task"]].extend(
            g for g in valid_geoms if not g.is_empty and g.area > 0
        )
        provenance.append(
            {
                "scene_id": scene["scene_id"],
                "chip_id": scene.get("chip_id"),
                "task": scene["task"],
                "source": scene.get("source", manifest.get("source", {})),
                "training_example": bool(scene.get("training_example", False)),
                "model": result.get("provenance", {}),
            }
        )
        if scene["task"] == "af":
            score = result.get("score")
            score_array = np.asarray(score) if score is not None else None
            if score_array is not None and score_array.shape not in {
                cmap.shape,
                (1, *cmap.shape),
                (2, *cmap.shape),
            }:
                raise ValueError(f"invalid score shape for {scene['scene_id']}")
            shared_grid = all(
                k in scene
                for k in ("source_scene_id", "grid_row_offset", "grid_col_offset")
            )
            candidates = valid if shared_grid else valid & (cmap == 1)
            for row, col in zip(*np.where(candidates)):
                cell = (
                    (
                        str(scene["source_scene_id"]),
                        scene.get("observed_at"),
                        int(scene["grid_row_offset"]) + int(row),
                        int(scene["grid_col_offset"]) + int(col),
                    )
                    if shared_grid
                    else (scene["scene_id"], int(row), int(col))
                )
                context = min(
                    int(row) + 1,
                    int(col) + 1,
                    cmap.shape[0] - int(row),
                    cmap.shape[1] - int(col),
                )
                priority = (-context, str(scene["scene_id"]))
                previous = af_cells.get(cell)
                if previous is not None and previous[0] <= priority:
                    continue
                if cmap[row, col] == 0:
                    af_cells[cell] = (priority, None)
                    continue
                x, y = xy(transform_affine, int(row), int(col), offset="center")
                point = shape(_to_wgs84(mapping(Point(x, y)), crs))
                if not aoi.covers(point):
                    continue
                value = (
                    None
                    if score_array is None
                    else float(
                        score_array[
                            (
                                1
                                if score_array.ndim == 3 and score_array.shape[0] > 1
                                else 0,
                                row,
                                col,
                            )
                        ]
                        if score_array.ndim == 3
                        else score_array[row, col]
                    )
                )
                af_cells[cell] = (
                    priority,
                    {
                        "type": "Feature",
                        "geometry": mapping(point),
                        "properties": {
                            "hotspot_id": ":".join(map(str, cell)),
                            "scene_id": scene["scene_id"],
                            "observed_at": scene.get("observed_at"),
                            "model_bundle_id": bundle_id,
                            "score": value
                            if value is not None and np.isfinite(value)
                            else None,
                        },
                    },
                )
        else:
            bs_records.append((scene, cmap, valid, transform_affine, crs))
    hotspots = [
        record[1]
        for _, record in sorted(af_cells.items(), key=lambda item: str(item[0]))
        if record[1] is not None
    ]
    # A later valid observation hides every older class, including older fire.
    occupied = None
    ordered_bs = sorted(bs_records, key=lambda x: x[0]["scene_id"])
    ordered_bs.sort(key=lambda x: x[0]["date_post"], reverse=True)
    for scene, cmap, valid, transform_affine, crs in ordered_bs:
        by_class = {klass: [] for klass in (1, 2, 3)}
        for geom, klass in shapes(
            cmap, mask=valid & (cmap > 0), transform=transform_affine, connectivity=4
        ):
            klass = int(klass)
            if klass not in (1, 2, 3):
                continue
            by_class[klass].append(shape(_to_wgs84(geom, crs)))
        claimed = None
        emitted = []
        # Different polygon boundaries can gain microscopic overlap when their
        # native-grid edges are independently projected. Resolve it explicitly
        # in output coordinates; severity 3, then 2, then 1 wins such slivers.
        for klass in (3, 2, 1):
            visible = unary_union(by_class[klass]).intersection(aoi)
            if occupied is not None:
                visible = visible.difference(occupied)
            if claimed is not None:
                visible = visible.difference(claimed)
            if visible.is_empty or visible.area == 0:
                continue
            claimed = visible if claimed is None else unary_union([claimed, visible])
            for part in _polygon_parts(visible):
                emitted.append(part)
                area = _area_ha(mapping(part), "EPSG:4326")
                unrounded_areas[str(klass)].append(area)
                contours.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(part),
                        "properties": {
                            "contour_id": f"{scene['scene_id']}:{klass}:{len(contours)}",
                            "class_id": klass,
                            "severity": {1: "low", 2: "medium", 3: "high"}[klass],
                            "area_ha": round(area, 6),
                            "date_pre": scene.get("date_pre"),
                            "date_post": scene.get("date_post"),
                            "source_scene_ids": [scene["scene_id"]],
                            "model_bundle_id": bundle_id,
                        },
                    }
                )
        current = [
            shape(_to_wgs84(g, crs)).intersection(aoi)
            for g, _ in shapes(
                valid.astype("uint8"),
                mask=valid,
                transform=transform_affine,
                connectivity=4,
            )
        ]
        current = [g for g in current if not g.is_empty and g.area > 0]
        current.extend(emitted)
        if current:
            occupied = (
                unary_union(current)
                if occupied is None
                else unary_union([occupied, *current])
            )
    from math import fsum

    areas = {klass: fsum(values) for klass, values in unrounded_areas.items()}
    observed = {
        task: min(aoi_area, _area_ha(mapping(unary_union(items)), "EPSG:4326"))
        if items
        else 0.0
        for task, items in coverage.items()
    }
    module_cov = {
        task: (observed[task] / aoi_area if aoi_area else 0.0) for task in coverage
    }
    task_count = {
        task: sum(1 for scene, _ in predictions if scene["task"] == task)
        for task in coverage
    }
    requested_tasks = requested_tasks or sorted(
        {scene["task"] for scene, _ in predictions}
    )
    modules, quality = {}, {}
    for task in requested_tasks:
        status = (
            "no_data"
            if observed[task] == 0
            else ("complete" if module_cov[task] >= 1 - 1e-8 else "partial")
        )
        modules[task] = {"data_status": status}
        if task == "af":
            modules[task]["hotspot_count"] = (
                None if status == "no_data" else len(hotspots)
            )
        else:
            modules[task].update(
                {
                    "burn_area_ha": None
                    if status == "no_data"
                    else sum(areas.values()),
                    "severity_area_ha": None if status == "no_data" else areas,
                }
            )
        quality[task] = {
            "observed_area_ha": observed[task],
            "unobserved_area_ha": max(0.0, aoi_area - observed[task]),
            "coverage_fraction": module_cov[task],
            "cloud_fraction": None,
            "observation_count": task_count[task],
        }
    _write_json(
        output / "hotspots.geojson", {"type": "FeatureCollection", "features": hotspots}
    )
    _write_json(
        output / "contours.geojson", {"type": "FeatureCollection", "features": contours}
    )
    _write_json(
        output / "provenance.json",
        {
            "model_bundle_id": bundle_id,
            "dataset_source": manifest.get("source", {}),
            "scenes": provenance,
        },
    )
    _write_json(output / "pixel_rles.json", pixel_rles)
    summary = {
        "modules": modules,
        "quality": {**quality, "warnings": []},
        "hotspot_count": modules.get("af", {}).get("hotspot_count"),
        "burn_area_ha": modules.get("bs", {}).get("burn_area_ha"),
        "severity_area_ha": modules.get("bs", {}).get("severity_area_ha"),
    }
    _write_json(output / "summary.json", summary)
    artifacts = {
        "summary": "summary.json",
        "hotspots": "hotspots.geojson",
        "burn_polygons": "contours.geojson",
        "provenance": "provenance.json",
        "pixel_rles": "pixel_rles.json",
        **{f"mask_{k}": v for k, v in masks.items()},
    }
    return {"summary": summary, "artifacts": artifacts, "pixel_rles": pixel_rles}


def run_analysis(
    request: dict[str, Any], output_dir: Path, data_root: Path, model_root: Path
) -> dict[str, Any]:
    """Run only selected georeferenced demo scenes and return JSON-safe references."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_for_request(request, Path(data_root), Path(model_root))
    scenes = (
        manifest["scenes"]
        if "_catalog_snapshot" in request
        else _selected_scenes(request, manifest)
    )
    bundle_id = request.get("model_bundle_id") or "official-v1"
    predictions = []
    if scenes:
        ready, _ = _model_ready(Path(model_root), bundle_id)
        if not ready:
            raise RuntimeError(f"requested model bundle is not ready: {bundle_id}")
        from competition.service_bridge import predict_chip

        predictions = [
            (
                scene,
                predict_chip(
                    manifest["_path"].parent,
                    str(scene["chip_id"]),
                    bundle_id=bundle_id,
                    model_dir=Path(model_root),
                ),
            )
            for scene in scenes
        ]
    tasks = request.get("tasks", [])
    exports = _prediction_exports(
        predictions, manifest, _request_aoi(request), output, bundle_id, tasks
    )
    result = exports.pop("summary")
    states = {value["data_status"] for value in result["modules"].values()}
    overall = (
        "no_data"
        if states == {"no_data"}
        else ("complete" if states == {"complete"} else "partial")
    )
    warnings = (
        ["No valid observations in the requested AOI and period"]
        if overall == "no_data"
        else []
    )
    result["quality"]["warnings"] = warnings
    provenance = json.loads((output / "provenance.json").read_text())
    provenance.update(
        {
            "request": {k: v for k, v in request.items() if not k.startswith("_")},
            "catalog_revision": request.get("_catalog_revision"),
            "model_manifest_sha256": request.get("_model_manifest_sha256"),
            "selected_scene_ids": [s["scene_id"] for s in scenes],
        }
    )
    _write_json(output / "provenance.json", provenance)
    result.update(
        {
            "schema_version": "1.0",
            "data_status": overall,
            "dataset_id": manifest["dataset_id"],
            "model_bundle_id": bundle_id,
            "tasks": tasks,
            "provenance": provenance,
            "warnings": warnings,
        }
    )
    _write_json(output / "summary.json", result)
    return {"result": result, "artifacts": exports["artifacts"], "data_status": overall}


def run_prediction(
    request: dict[str, Any], output_dir: Path, data_root: Path, model_root: Path
) -> dict[str, Any]:
    """Raw registered-chip products with C-order competition RLE."""
    chip_ids = request.get("chip_ids") or (
        [request["chip_id"]] if request.get("chip_id") else []
    )
    if not isinstance(chip_ids, list) or not chip_ids:
        raise ValueError("chip_ids must contain at least one registered chip id")
    manifest = _manifest_for_request(request, Path(data_root), Path(model_root))
    scenes = {
        str(s.get("chip_id")): s
        for s in manifest.get("scenes", [])
        if isinstance(s, dict)
    }
    requested = [str(x) for x in chip_ids]
    missing = [x for x in requested if x not in scenes]
    if missing:
        raise FileNotFoundError(f"unknown registered chip ids: {', '.join(missing)}")
    bundle = str(request.get("model_bundle_id") or "official-v1")
    if not _model_ready(Path(model_root), bundle)[0]:
        raise RuntimeError(f"requested model bundle is not ready: {bundle}")
    from competition.service_bridge import predict_chip
    from competition.rle import encode_bs_rles, encode_rle
    import rasterio

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    base = manifest["_path"].parent
    items = []
    artifacts = {}
    provenance = []
    for chip_id in requested:
        scene = scenes[chip_id]
        pred = predict_chip(base, chip_id, bundle_id=bundle, model_dir=Path(model_root))
        raw_cmap = np.asarray(pred["class_map"])
        valid = np.asarray(pred["observation_valid"], dtype=bool)
        allowed = [0, 1] if scene["task"] == "af" else [0, 1, 2, 3]
        if (
            raw_cmap.ndim != 2
            or raw_cmap.shape != valid.shape
            or not np.issubdtype(raw_cmap.dtype, np.number)
            or not np.isin(raw_cmap, allowed).all()
        ):
            raise ValueError(f"invalid raw prediction for {chip_id}")
        cmap = raw_cmap.astype(np.uint8)
        aid = f"mask_{chip_id}"
        ref = scene.get("reference_raster")
        geo = bool(ref)
        if geo:
            transform_affine, crs = _transform_for(scene, base)
            filename = f"{aid}.tif"
            with rasterio.open(base / ref) as reference:
                if (reference.height, reference.width) != cmap.shape:
                    raise ValueError(
                        f"prediction dimensions differ from reference raster for {chip_id}"
                    )
            with rasterio.open(
                root / filename,
                "w",
                driver="GTiff",
                height=cmap.shape[0],
                width=cmap.shape[1],
                count=1,
                dtype="uint8",
                crs=crs,
                transform=transform_affine,
                compress="deflate",
            ) as dst:
                dst.write(cmap, 1)
        else:
            filename = f"{aid}.npy"
            np.save(root / filename, cmap)
        artifacts[aid] = filename
        rles = (
            {"1": encode_rle(cmap == 1)}
            if scene["task"] == "af"
            else {str(k): v for k, v in encode_bs_rles(cmap).items()}
        )
        items.append(
            {
                "chip_id": chip_id,
                "task": scene["task"],
                "height": int(cmap.shape[0]),
                "width": int(cmap.shape[1]),
                "georeferenced": geo,
                "classes": allowed,
                "mask_artifact_id": aid,
                "rle": [{"class_id": int(k), "rle": v} for k, v in rles.items()],
            }
        )
        provenance.append(
            {
                "chip_id": chip_id,
                "scene_id": scene.get("scene_id"),
                "model": pred.get("provenance", {}),
            }
        )
    _write_json(
        root / "prediction_provenance.json",
        {
            "model_bundle_id": bundle,
            "items": provenance,
            "catalog_revision": request.get("_catalog_revision"),
            "model_manifest_sha256": request.get("_model_manifest_sha256"),
        },
    )
    artifacts["prediction_provenance"] = "prediction_provenance.json"
    return {
        "items": items,
        "artifacts": artifacts,
        "provenance": {
            "model_bundle_id": bundle,
            "dataset_source": manifest.get("source", {}),
        },
    }
