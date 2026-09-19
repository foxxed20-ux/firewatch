"""Discovery for the geospatial demo manifest.

The served demo is deliberately opt-in: ``DATA_ROOT/datasets/<id>/manifest.json``
is the only supported source of scenes.  A manifest has ``dataset_id``, ``source``
and ``scenes``; every scene records an id, task, dates and a georeferenced
``reference_raster`` (or explicit ``crs``/``transform``/``width``/``height``).
Competition chips are never discovered or published by this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def discover_datasets(data_root: Path) -> list[dict[str, Any]]:
    """Return validated local manifests; malformed files are not exposed."""
    base = Path(data_root) / "datasets"
    if not base.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(base.glob("*/manifest.json")):
        manifest = _read_json(path)
        if not manifest or not isinstance(manifest.get("dataset_id"), str):
            continue
        if manifest.get("mode", "geospatial_demo") != "geospatial_demo":
            continue
        manifest["_path"] = path
        found.append(manifest)
    return found


def _scene_dates(scene: dict[str, Any]) -> list[str]:
    return [
        str(scene[k]) for k in ("observed_at", "date_pre", "date_post") if scene.get(k)
    ]


def _bounds(manifest: dict[str, Any]) -> list[float] | None:
    value = manifest.get("bounds")
    if (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(x, (int, float)) for x in value)
    ):
        return [float(x) for x in value]
    return None


def describe_catalog(data_root: Path) -> dict[str, Any]:
    """Public catalog payload.  It contains no paths and no invented scenes."""
    datasets: list[dict[str, Any]] = []
    for manifest in discover_datasets(Path(data_root)):
        scenes = (
            manifest.get("scenes") if isinstance(manifest.get("scenes"), list) else []
        )
        dates = sorted(
            d for s in scenes if isinstance(s, dict) for d in _scene_dates(s)
        )
        preset_ends = [
            str(p.get("period", {}).get("end"))
            for p in manifest.get("presets", [])
            if isinstance(p, dict) and p.get("period", {}).get("end")
        ]
        tasks = sorted(
            {
                str(s.get("task"))
                for s in scenes
                if isinstance(s, dict) and s.get("task") in {"af", "bs"}
            }
        )
        datasets.append(
            {
                "dataset_id": manifest["dataset_id"],
                "label": str(manifest.get("label", manifest["dataset_id"])),
                "mode": "geospatial_demo",
                "bounds": _bounds(manifest),
                "date_range": {
                    "start": dates[0] if dates else None,
                    "end": max([*dates, *preset_ends])
                    if (dates or preset_ends)
                    else None,
                },
                "tasks": tasks,
                "scene_count": len(scenes),
                "source": manifest.get("source", {}),
                "source_assets": manifest.get("source_assets", []),
                "presets": manifest.get("presets", []),
            }
        )
    return {"datasets": datasets}


def describe_models(model_root: Path) -> dict[str, Any]:
    """Expose bundle state without claiming readiness before the bridge does."""
    try:
        from competition.service_bridge import describe_models as bridge_describe

        raw = bridge_describe(model_dir=Path(model_root))
    except Exception as exc:  # Bundle may be training or intentionally absent.
        manifest_path = Path(model_root) / "manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.is_file() else None
        raw = (
            {
                **manifest,
                "available": False,
                "bridge_warning": f"model bridge unavailable: {type(exc).__name__}",
            }
            if manifest
            else {
                "models": [],
                "error": f"model bridge unavailable: {type(exc).__name__}",
            }
        )
    if isinstance(raw, list):
        raw = {"models": raw}
    if not isinstance(raw, dict):
        raw = {"models": []}
    # Bridge v1 returns one manifest object; retain support for a future list.
    if "tasks" in raw and ("bundle_id" in raw or "model_bundle_id" in raw):
        models = [raw]
    else:
        models = raw.get("models", raw.get("bundles", []))
    if not isinstance(models, list):
        models = []
    normalized = []
    for item in models:
        if not isinstance(item, dict):
            continue
        tasks = item.get("tasks", {})
        # A successfully imported bridge is the sole artifact verifier.  Its
        # manifest supports multiple backend layouts (including ensembles).
        integrity = bool(item.get("available", item.get("status") == "ready"))
        normalized.append(
            {
                "model_bundle_id": item.get(
                    "model_bundle_id", item.get("bundle_id", "official-v1")
                ),
                "status": "ready" if integrity else "unavailable",
                "tasks": list(tasks) if isinstance(tasks, dict) else tasks,
                "provenance": item.get("provenance", {}),
                "warnings": item.get("warnings", []),
            }
        )
    return {
        "models": normalized,
        **({"error": raw["error"]} if raw.get("error") else {}),
    }
