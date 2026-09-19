import json
import sys
import types
import importlib.util
import zipfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

from firewatch_service.catalog import describe_catalog
from firewatch_service.catalog import discover_datasets
from firewatch_service.processing import _selected_scenes, run_analysis, run_prediction


def _dataset(tmp_path):
    root = tmp_path / "data"
    folder = root / "datasets" / "demo"
    folder.mkdir(parents=True)
    with rasterio.open(
        folder / "reference.tif",
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(30, 50, 0.01, 0.01),
    ) as dst:
        dst.write(np.zeros((1, 2, 2), dtype=np.uint8))
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "demo",
                "mode": "geospatial_demo",
                "bounds": [30, 49.98, 30.02, 50],
                "source": {"kind": "fixture"},
                "scenes": [
                    {
                        "scene_id": "af-scene",
                        "chip_id": "chip-1",
                        "task": "af",
                        "observed_at": "2024-07-01T00:00:00Z",
                        "reference_raster": "reference.tif",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_catalog_only_exposes_manifest_datasets(tmp_path):
    root = _dataset(tmp_path)
    catalog = describe_catalog(root)
    assert catalog["datasets"][0]["dataset_id"] == "demo"
    assert catalog["datasets"][0]["source"] == {"kind": "fixture"}


def test_scene_period_is_af_observed_and_bs_post_half_open(tmp_path):
    root = _dataset(tmp_path)
    manifest = discover_datasets(root)[0]
    request = {
        "tasks": ["af"],
        "aoi": {"bbox": [29, 49, 31, 51]},
        "period": {"start": "2024-06-01T00:00:00Z", "end": "2024-07-01T00:00:00Z"},
    }
    assert _selected_scenes(request, manifest) == []  # observed_at equals exclusive end
    manifest["scenes"][0].update(
        {
            "task": "bs",
            "date_pre": "2024-06-01T00:00:00Z",
            "date_post": "2024-07-01T00:00:00Z",
        }
    )
    request["tasks"] = ["bs"]
    assert _selected_scenes(request, manifest) == []  # post equals exclusive end


def test_analysis_writes_valid_mask_and_hotspots(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    bridge = types.ModuleType("competition.service_bridge")
    bridge.describe_models = lambda model_dir=None: {
        "models": [
            {
                "bundle_id": "official-v1",
                "status": "ready",
                "tasks": ["af"],
                "provenance": {"test": True},
            }
        ]
    }
    bridge.predict_chip = lambda *args, **kwargs: {
        "class_map": np.array([[1, 0], [1, 1]], dtype=np.uint8),
        "observation_valid": np.array([[True, False], [True, True]]),
        "score": np.array([[0.8, np.nan], [0.7, 0.6]], dtype=np.float32),
        "provenance": {},
    }
    monkeypatch.setitem(sys.modules, "competition.service_bridge", bridge)
    out = tmp_path / "out"
    result = run_analysis(
        {
            "dataset_id": "demo",
            "model_bundle_id": "official-v1",
            "tasks": ["af"],
            "aoi": {"bbox": [29.9, 49.9, 30.1, 50.1]},
            "period": {"start": "2024-06-01T00:00:00Z", "end": "2024-08-01T00:00:00Z"},
        },
        out,
        root,
        tmp_path / "models",
    )
    assert result["data_status"] in {"complete", "partial"}
    assert set(result["result"]) >= {"modules", "quality"}
    # A task not requested is omitted, rather than reported as unobserved.
    assert set(result["result"]["quality"]) == {"af", "warnings"}
    assert set(result["result"]["modules"]) == {"af"}
    assert "burn_polygons" in result["artifacts"]
    assert result["result"]["hotspot_count"] == 3
    with rasterio.open(out / "mask_af-scene.tif") as src:
        assert src.nodata == 255
        assert src.read(1)[0, 1] == 255
    assert len(json.loads((out / "hotspots.geojson").read_text())["features"]) == 3


def test_prediction_accepts_rest_chip_ids(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    bridge = types.ModuleType("competition.service_bridge")
    bridge.describe_models = lambda model_dir=None: {
        "models": [{"bundle_id": "official-v1", "status": "ready"}]
    }
    bridge.predict_chip = lambda *args, **kwargs: {
        "class_map": np.array([[0, 0], [0, 0]], dtype=np.uint8),
        "observation_valid": np.ones((2, 2), dtype=bool),
        "provenance": {},
    }
    monkeypatch.setitem(sys.modules, "competition.service_bridge", bridge)
    result = run_prediction(
        {
            "dataset_id": "demo",
            "model_bundle_id": "official-v1",
            "chip_ids": ["chip-1"],
        },
        tmp_path / "out",
        root,
        tmp_path / "models",
    )
    assert result["items"][0]["chip_id"] == "chip-1"
    assert result["items"][0]["height"] == 2
    assert (tmp_path / "out" / "mask_chip-1.tif").is_file()


def test_bs_exports_clipped_severity_area(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    manifest_path = root / "datasets" / "demo" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scenes"] = [
        {
            "scene_id": "bs-scene",
            "chip_id": "chip-bs",
            "task": "bs",
            "date_pre": "2024-06-01T00:00:00Z",
            "date_post": "2024-07-01T00:00:00Z",
            "reference_raster": "reference.tif",
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bridge = types.ModuleType("competition.service_bridge")
    bridge.describe_models = lambda model_dir=None: {
        "models": [{"bundle_id": "official-v1", "status": "ready"}]
    }
    bridge.predict_chip = lambda *args, **kwargs: {
        "class_map": np.array([[1, 2], [3, 0]], dtype=np.uint8),
        "observation_valid": np.ones((2, 2), dtype=bool),
        "provenance": {},
    }
    monkeypatch.setitem(sys.modules, "competition.service_bridge", bridge)
    result = run_analysis(
        {
            "dataset_id": "demo",
            "model_bundle_id": "official-v1",
            "tasks": ["bs"],
            "aoi": {"bbox": [30, 49.99, 30.01, 50]},
            "period": {"start": "2024-06-01T00:00:00Z", "end": "2024-08-01T00:00:00Z"},
        },
        tmp_path / "out",
        root,
        tmp_path / "models",
    )
    areas = result["result"]["severity_area_ha"]
    assert result["result"]["burn_area_ha"] == sum(areas.values())
    assert areas["1"] > 0 and areas["2"] == 0 and areas["3"] == 0
    assert (
        len(json.loads((tmp_path / "out" / "contours.geojson").read_text())["features"])
        == 1
    )


def test_bs_latest_valid_observation_hides_older_overlap(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    manifest_path = root / "datasets" / "demo" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scenes"] = [
        {
            "scene_id": "old",
            "chip_id": "old",
            "task": "bs",
            "date_pre": "2024-06-01T00:00:00Z",
            "date_post": "2024-07-01T00:00:00Z",
            "reference_raster": "reference.tif",
        },
        {
            "scene_id": "new",
            "chip_id": "new",
            "task": "bs",
            "date_pre": "2024-07-01T00:00:00Z",
            "date_post": "2024-08-01T00:00:00Z",
            "reference_raster": "reference.tif",
        },
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bridge = types.ModuleType("competition.service_bridge")
    bridge.describe_models = lambda model_dir=None: {
        "models": [{"bundle_id": "official-v1", "status": "ready"}]
    }
    bridge.predict_chip = lambda dataset, chip_id, **kwargs: {
        "class_map": np.full((2, 2), 1 if chip_id == "old" else 3, dtype=np.uint8),
        "observation_valid": np.ones((2, 2), dtype=bool),
        "provenance": {},
    }
    monkeypatch.setitem(sys.modules, "competition.service_bridge", bridge)
    result = run_analysis(
        {
            "dataset_id": "demo",
            "model_bundle_id": "official-v1",
            "tasks": ["bs"],
            "aoi": {"bbox": [29, 49, 31, 51]},
            "period": {"start": "2024-06-01T00:00:00Z", "end": "2024-09-01T00:00:00Z"},
        },
        tmp_path / "out",
        root,
        tmp_path / "models",
    )
    areas = result["result"]["severity_area_ha"]
    assert areas["1"] == 0 and areas["3"] > 0


def test_demo_import_rejects_archive_traversal(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_service_demo", "tools/prepare_service_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.tif", b"x")
    with zipfile.ZipFile(archive_path) as archive:
        import pytest

        with pytest.raises(ValueError, match="unsafe ZIP member"):
            module._safe_members(archive)


def _request():
    return {
        "dataset_id": "demo",
        "model_bundle_id": "official-v1",
        "tasks": ["af", "bs"],
        "aoi": {"bbox": [29, 49, 31, 51]},
        "period": {"start": "2024-06-01T00:00:00Z", "end": "2024-09-01T00:00:00Z"},
    }


def _bridge(monkeypatch, valid=True):
    bridge = types.ModuleType("competition.service_bridge")
    bridge.describe_models = lambda model_dir=None: {
        "models": [{"bundle_id": "official-v1", "status": "ready"}]
    }
    bridge.predict_chip = lambda *args, **kwargs: {
        "class_map": np.ones((2, 2), dtype=np.uint8),
        "observation_valid": np.full((2, 2), valid, dtype=bool),
        "provenance": {},
    }
    monkeypatch.setitem(sys.modules, "competition.service_bridge", bridge)
    return bridge


def test_no_scenes_and_all_invalid_have_downloadable_coverage(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    _bridge(monkeypatch, valid=False)
    for name, period, count in [
        ("invalid", _request()["period"], 1),
        (
            "no-scene",
            {"start": "2025-01-01T00:00:00Z", "end": "2025-02-01T00:00:00Z"},
            0,
        ),
    ]:
        request = {**_request(), "period": period}
        out = tmp_path / name
        produced = run_analysis(request, out, root, tmp_path / "models")
        result = produced["result"]
        assert produced["data_status"] == "no_data"
        for task in request["tasks"]:
            assert result["modules"][task]["data_status"] == "no_data"
            assert result["quality"][task]["coverage_fraction"] == 0
            assert result["quality"][task]["unobserved_area_ha"] > 0
        assert result["modules"]["af"]["hotspot_count"] is None
        assert result["modules"]["bs"]["burn_area_ha"] is None
        assert result["quality"]["af"]["observation_count"] == count
        assert (
            json.loads((out / produced["artifacts"]["summary"]).read_text()) == result
        )
        assert (out / produced["artifacts"]["provenance"]).is_file()


def test_missing_requested_bs_is_no_data_but_af_is_observed(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    _bridge(monkeypatch)
    result = run_analysis(_request(), tmp_path / "out", root, tmp_path / "models")[
        "result"
    ]
    assert result["data_status"] == "partial"
    assert result["modules"]["af"]["hotspot_count"] == 4
    assert result["modules"]["bs"]["burn_area_ha"] is None
    assert result["quality"]["bs"]["observation_count"] == 0


def test_snapshot_preserves_scene_selection_after_manifest_edit(tmp_path, monkeypatch):
    from firewatch_service.processing import snapshot_request

    root = _dataset(tmp_path)
    _bridge(monkeypatch)
    request = snapshot_request("analysis", _request(), root, tmp_path / "models")
    path = root / "datasets" / "demo" / "manifest.json"
    changed = json.loads(path.read_text())
    changed["scenes"] = []
    path.write_text(json.dumps(changed))
    result = run_analysis(request, tmp_path / "out", root, tmp_path / "models")[
        "result"
    ]
    assert result["modules"]["af"]["hotspot_count"] == 4
    assert result["provenance"]["selected_scene_ids"] == ["af-scene"]
    assert len(result["provenance"]["catalog_revision"]) == 64


def test_af_same_source_grid_is_deduplicated_with_background_winner(
    tmp_path, monkeypatch
):
    root = _dataset(tmp_path)
    path = root / "datasets" / "demo" / "manifest.json"
    manifest = json.loads(path.read_text())
    original = manifest["scenes"][0]
    manifest["scenes"] = [
        {
            **original,
            "scene_id": name,
            "chip_id": name,
            "source_scene_id": "same-overpass",
            "grid_row_offset": 0,
            "grid_col_offset": 0,
        }
        for name in ["b", "a"]
    ]
    path.write_text(json.dumps(manifest))
    bridge = _bridge(monkeypatch)
    bridge.predict_chip = lambda dataset, chip_id, **kwargs: {
        "class_map": np.full((2, 2), 0 if chip_id == "a" else 1, dtype=np.uint8),
        "observation_valid": np.ones((2, 2), dtype=bool),
        "provenance": {},
    }
    result = run_analysis(_request(), tmp_path / "out", root, tmp_path / "models")[
        "result"
    ]
    assert result["modules"]["af"]["hotspot_count"] == 0
