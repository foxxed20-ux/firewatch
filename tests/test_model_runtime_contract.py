import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_source(monkeypatch, name: str, path: Path, stubs: dict[str, ModuleType]):
    for module_name, module in stubs.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_official_template_produces_all_447_quoted_rows(tmp_path, monkeypatch) -> None:
    af_ids = [f"AF_te_{index:06d}" for index in range(1, 181)]
    bs_ids = [f"BS_te_{index:06d}" for index in range(1, 90)]
    expected_keys = [(chip_id, 1) for chip_id in af_ids]
    expected_keys += [
        (chip_id, class_id)
        for chip_id in bs_ids
        for class_id in (1, 2, 3)
    ]
    assert len(expected_keys) == 447

    template = tmp_path / "sample_submission.csv"
    with template.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("chip_id", "class_id", "rle"))
        writer.writerows((chip_id, class_id, "") for chip_id, class_id in expected_keys)

    inputs = {chip_id: {} for chip_id in (*af_ids, *bs_ids)}
    data_stub = ModuleType("competition.data")
    data_stub.discover_chips = lambda _root: inputs
    data_stub.read_chip = lambda chip_id, _paths: SimpleNamespace(
        chip_id=chip_id,
        task="af" if chip_id.startswith("AF_") else "bs",
    )
    bridge_stub = ModuleType("competition.service_bridge")

    def predict_features(chip, _model_dir):
        mask = np.zeros((2, 3), dtype=np.uint8)
        if chip.chip_id == af_ids[0]:
            mask[:] = ((0, 1, 1), (0, 0, 1))
        elif chip.chip_id == bs_ids[0]:
            mask[:] = ((1, 2, 3), (0, 0, 0))
        return {"class_map": mask}

    bridge_stub.predict_features = predict_features
    bridge_stub.describe_models = lambda _model_dir: {"bundle_id": "contract-test"}
    inference = _load_source(
        monkeypatch,
        "_firewatch_inference_contract",
        PROJECT_ROOT / "inference.py",
        {
            "competition.data": data_stub,
            "competition.service_bridge": bridge_stub,
        },
    )

    output = tmp_path / "submission.csv"
    assert inference.main(
        ["--data-dir", str(tmp_path), "--output", str(output)]
    ) == 0

    with output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(row["chip_id"], int(row["class_id"])) for row in rows] == expected_keys
    assert rows[0]["rle"] == "2 2 6 1"
    assert len(rows) == 447

    text = output.read_text(encoding="utf-8")
    assert text.splitlines()[0] == '"chip_id","class_id","rle"'
    assert any(line.endswith(',""') for line in text.splitlines()[1:])
    audit = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
    assert audit["rows"] == 447
    assert audit["chips"] == 269
    assert audit["round_trip_verified"] is True


def test_runtime_requires_feature_order_and_scores_every_pixel(tmp_path, monkeypatch) -> None:
    data_stub = ModuleType("competition.data")
    data_stub.FEATURE_VERSION = "official-named-v1"
    data_stub.discover_chips = lambda _root: {}
    data_stub.read_chip = lambda *_args: None
    bridge = _load_source(
        monkeypatch,
        "competition._service_bridge_contract",
        PROJECT_ROOT / "competition" / "service_bridge.py",
        {"competition.data": data_stub},
    )

    model_root = tmp_path / "bundle"
    model_root.mkdir()
    (model_root / "af").mkdir()
    (model_root / "bs").mkdir()
    incomplete_manifest = {
        "bundle_id": "broken",
        "feature_version": "official-named-v1",
        "tasks": {
            "af": {"backend": "lightgbm", "path": "af", "sha256": {}},
            "bs": {"backend": "torch", "path": "bs", "sha256": {}},
        },
    }
    (model_root / "manifest.json").write_text(
        json.dumps(incomplete_manifest), encoding="utf-8"
    )
    description = bridge.describe_models(model_root)
    assert description["available"] is False
    assert "artifact hashes" in description["reason"]

    (model_root / "manifest.json").write_text("{}", encoding="utf-8")

    class Booster:
        def predict(self, features, num_threads):
            assert num_threads == 2
            return np.full(len(features), 0.9, dtype=np.float32)

    manifest = {"bundle_id": "contract-test"}
    spec = {"backend": "lightgbm", "sha256": {"model.txt": "digest"}}
    metadata = {"feature_names": ["I1"], "threshold": 0.5}
    monkeypatch.setattr(
        bridge,
        "_load",
        lambda *_args: (manifest, spec, Booster(), metadata, "cpu"),
    )
    chip = SimpleNamespace(
        task="af",
        feature_names=["I1"],
        x=np.zeros((1, 2, 2), dtype=np.float32),
        observation_valid=np.array([[True, False], [False, True]]),
    )

    result = bridge.predict_features(chip, model_root)
    np.testing.assert_array_equal(result["class_map"], np.ones((2, 2), dtype=np.uint8))
    np.testing.assert_array_equal(result["observation_valid"], chip.observation_valid)
    assert "never an output mask" in result["provenance"]["cloud_policy"]

    chip.feature_names = ["wrong"]
    with pytest.raises(ValueError, match="Ordered feature names"):
        bridge.predict_features(chip, model_root)


def test_preparation_rejects_a_split_with_a_missing_chip(tmp_path, monkeypatch) -> None:
    data_stub = ModuleType("competition.data")
    data_stub.FEATURE_VERSION = "official-named-v1"
    data_stub.discover_chips = lambda _root: {"AF_tr_000001": {}}

    def unexpected_read(*_args):
        raise AssertionError("preparation must validate the inventory before reading chips")

    data_stub.read_chip = unexpected_read
    data_stub.read_training_mask = unexpected_read
    prepare = _load_source(
        monkeypatch,
        "competition._prepare_contract",
        PROJECT_ROOT / "competition" / "prepare.py",
        {"competition.data": data_stub},
    )
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"AF_tr_000001": "train", "AF_tr_000002": "val"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Split/input mismatch"):
        prepare.main(
            [
                "--data-dir",
                str(tmp_path / "train"),
                "--output",
                str(tmp_path / "prepared"),
                "--split",
                str(split),
            ]
        )


def test_bs_ensemble_blends_probabilities_then_applies_selected_multipliers(
    tmp_path, monkeypatch
) -> None:
    data_stub = ModuleType("competition.data")
    data_stub.FEATURE_VERSION = "official-named-v1"
    data_stub.discover_chips = lambda _root: {}
    data_stub.read_chip = lambda *_args: None
    bridge = _load_source(
        monkeypatch,
        "competition._service_bridge_ensemble_contract",
        PROJECT_ROOT / "competition" / "service_bridge.py",
        {"competition.data": data_stub},
    )

    model_root = tmp_path / "bundle"
    model_root.mkdir()
    (model_root / "manifest.json").write_text("{}", encoding="utf-8")
    components = {
        "cnn": {"backend": "torch", "path": "bs/cnn", "sha256": {}},
        "tree": {"backend": "lightgbm", "path": "bs/tree", "sha256": {}},
    }
    spec = {
        "backend": "ensemble",
        "components": components,
        "tree_weight": 0.25,
        "postprocess": {"class_multipliers": [1.0, 1.0, 2.0, 1.0]},
    }
    manifest = {"bundle_id": "ensemble-contract"}
    models = {"cnn": "cnn-model", "tree": "tree-model"}
    monkeypatch.setattr(
        bridge,
        "_load",
        lambda *_args: (
            manifest,
            spec,
            models,
            {"cnn": {}, "tree": {}},
            {"cnn": "cpu", "tree": "cpu"},
        ),
    )
    cnn_score = np.array([[[0.0]], [[0.8]], [[0.2]], [[0.0]]], dtype=np.float32)
    tree_score = np.array([[[0.0]], [[0.2]], [[0.8]], [[0.0]]], dtype=np.float32)
    monkeypatch.setattr(
        bridge,
        "_probabilities",
        lambda _chip, _component, model, _meta, _device: (
            cnn_score if model == "cnn-model" else tree_score
        ),
    )
    chip = SimpleNamespace(
        task="bs",
        feature_names=["feature"],
        x=np.zeros((1, 1, 1), dtype=np.float32),
        observation_valid=np.array([[False]]),
    )

    result = bridge.predict_features(chip, model_root)

    # Blend is [class1=.65, class2=.35]; the selected x2 class-2 multiplier
    # must then change the final decision from class 1 to class 2.
    assert result["score"][:, 0, 0] == pytest.approx([0.0, 0.65, 0.35, 0.0])
    assert result["class_map"].tolist() == [[2]]
    assert result["provenance"]["backend"] == "ensemble"
