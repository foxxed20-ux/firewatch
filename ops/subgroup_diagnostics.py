"""Post-hoc micro diagnostics for frozen validation predictions.

This tool never tunes thresholds or class weights. Cache order follows
competition.evaluate: load_records preserves manifest order, main filters that
ordered list by task/split, and _load_probability appends/np.stack's in order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(path: Path, task: str) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["records"] if isinstance(raw, dict) else raw
    selected = [
        row
        for row in rows
        if str(row["task"]).lower() == task and str(row["split"]) == "val"
    ]
    ids = [str(row["id"]) for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {task} validation record IDs")
    if not selected:
        raise ValueError(f"no {task} validation records")
    return selected


def _metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {str(row["chip_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate chip_id in {path}")
    return result


def _year(value: str, *, field: str, chip_id: str) -> str:
    match = re.match(r"^(\d{4})", value or "")
    if not match:
        raise ValueError(f"{chip_id}: invalid {field}={value!r}")
    return match.group(1)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _af_counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    tp = int(np.count_nonzero((truth == 1) & (prediction == 1)))
    fp = int(np.count_nonzero((truth == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((truth == 1) & (prediction == 0)))
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": 1.0 if denominator == 0 else float(2 * tp / denominator),
    }


def _confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if not np.isin(truth, (0, 1, 2, 3)).all():
        raise ValueError(f"BS truth has unexpected labels: {np.unique(truth)}")
    if not np.isin(prediction, (0, 1, 2, 3)).all():
        raise ValueError(
            f"BS prediction has unexpected labels: {np.unique(prediction)}"
        )
    return np.bincount(
        truth.astype(np.int64) * 4 + prediction.astype(np.int64), minlength=16
    ).reshape(4, 4)


def _bs_metrics(confusion: np.ndarray) -> dict[str, Any]:
    burn_tp = int(confusion[1:, 1:].sum())
    burn_fp = int(confusion[0, 1:].sum())
    burn_fn = int(confusion[1:, 0].sum())
    burn_union = burn_tp + burn_fp + burn_fn
    severity: dict[str, float] = {}
    for cls in (1, 2, 3):
        tp = int(confusion[cls, cls])
        union = int(confusion[cls, :].sum() + confusion[:, cls].sum() - tp)
        severity[str(cls)] = 1.0 if union == 0 else float(tp / union)
    return {
        "confusion_rows_truth_columns_prediction": confusion.astype(int).tolist(),
        "burn": {
            "tp": burn_tp,
            "fp": burn_fp,
            "fn": burn_fn,
            "precision": _ratio(burn_tp, burn_tp + burn_fp),
            "recall": _ratio(burn_tp, burn_tp + burn_fn),
            "iou": 1.0 if burn_union == 0 else float(burn_tp / burn_union),
        },
        "severity_iou": severity,
        "miou_severity": float(sum(severity.values()) / 3),
    }


def _group_indices(
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, str]],
    key: Callable[[dict[str, str], str], str],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        chip_id = str(row["id"])
        if chip_id not in metadata:
            raise ValueError(f"{chip_id}: missing metadata")
        groups[key(metadata[chip_id], chip_id)].append(index)
    return dict(sorted(groups.items()))


def _af_group(
    indices: list[int],
    reference: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    truth = np.concatenate([reference[i][valid[i]] for i in indices])
    pred = np.concatenate([prediction[i][valid[i]] for i in indices])
    return {
        "chips": len(indices),
        "scored_pixels": int(len(truth)),
        **_af_counts(truth, pred),
    }


def _bs_group(
    indices: list[int],
    reference: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    confusion = np.zeros((4, 4), dtype=np.int64)
    pixels = 0
    for index in indices:
        truth = reference[index][valid[index]]
        pred = prediction[index][valid[index]]
        confusion += _confusion(truth, pred)
        pixels += len(truth)
    return {
        "chips": len(indices),
        "scored_pixels": int(pixels),
        **_bs_metrics(confusion),
    }


def _cache(path: Path, expected_n: int, task: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"reference", "valid", "tree_probability"}
    if task == "af":
        required.add("cnn_probability")
    if not required.issubset(payload):
        raise ValueError(f"{path}: missing keys {sorted(required - payload.keys())}")
    if any(value.shape[0] != expected_n for value in payload.values()):
        shapes = {key: list(value.shape) for key, value in payload.items()}
        raise ValueError(f"{path}: record/cache order length mismatch: {shapes}")
    reference, valid = payload["reference"], payload["valid"]
    expected_hw = (256, 256) if task == "af" else (512, 512)
    if reference.shape != (expected_n, *expected_hw) or valid.shape != reference.shape:
        raise ValueError(f"{path}: unexpected reference/valid shapes")
    if valid.dtype != np.bool_:
        raise ValueError(f"{path}: valid must be bool")
    allowed = (0, 1) if task == "af" else (0, 1, 2, 3)
    if not np.isin(reference[valid], allowed).all():
        raise ValueError(f"{path}: unexpected reference labels")
    for key in ("cnn_probability", "tree_probability"):
        if key in payload and not np.isfinite(payload[key]).all():
            raise ValueError(f"{path}: non-finite {key}")
    return payload


def _bs_cnn_cache(path: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"reference", "valid", "cnn_probability"}
    if not required.issubset(payload):
        raise ValueError(f"{path}: missing keys {sorted(required - payload.keys())}")
    expected_n = len(rows)
    if (
        payload["reference"].shape != (expected_n, 512, 512)
        or payload["valid"].shape != payload["reference"].shape
    ):
        raise ValueError(f"{path}: unexpected BS reference/valid shapes")
    if payload["cnn_probability"].shape != (*payload["reference"].shape, 4):
        raise ValueError(f"{path}: unexpected BS cnn_probability shape")
    if (
        payload["valid"].dtype != np.bool_
        or not np.isfinite(payload["cnn_probability"]).all()
    ):
        raise ValueError(f"{path}: invalid valid/CNN probability values")
    if not np.isin(payload["reference"][payload["valid"]], (0, 1, 2, 3)).all():
        raise ValueError(f"{path}: unexpected BS reference labels")
    if "chip_ids" in payload:
        expected_ids = np.asarray([str(row["id"]) for row in rows])
        if not np.array_equal(payload["chip_ids"].astype(str), expected_ids):
            raise ValueError(f"{path}: chip_ids do not match stable records order")
    return payload


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    records_path = Path(args.records)
    recipes_path = Path(args.recipes)
    af_cache_path = Path(args.af_cache)
    bs_cache_path = Path(args.bs_cache)
    bs_cnn_cache_path = Path(args.bs_cnn_cache) if args.bs_cnn_cache else None
    af_rows = _records(records_path, "af")
    bs_rows = _records(records_path, "bs")
    af_meta = _metadata(Path(args.af_meta))
    bs_meta = _metadata(Path(args.bs_meta))
    af = _cache(af_cache_path, len(af_rows), "af")
    bs = _cache(bs_cache_path, len(bs_rows), "bs")
    recipes = json.loads(recipes_path.read_text(encoding="utf-8"))

    af_recipe = recipes["af"]
    selected = str(af_recipe["selected"])
    match = re.fullmatch(r"ensemble_cnn_(0(?:\.\d+)?|1(?:\.0+)?)", selected)
    if af_recipe.get("backend") != "ensemble" or not match:
        raise ValueError(f"unexpected AF V2 recipe: {af_recipe}")
    cnn_weight = float(match.group(1))
    af_threshold = float(af_recipe["threshold"])
    tree_weight = 1.0 - cnn_weight
    tree_af = af["tree_probability"]
    if tree_af.shape != (*af["reference"].shape, 1):
        raise ValueError(f"unexpected AF tree_probability shape {tree_af.shape}")
    if af["cnn_probability"].shape != af["reference"].shape:
        raise ValueError("unexpected AF cnn_probability shape")
    af_probability = cnn_weight * af["cnn_probability"] + tree_weight * tree_af[..., 0]
    af_prediction = (af_probability >= af_threshold).astype(np.uint8)

    bs_recipe = recipes["bs"]
    bs_backend = bs_recipe.get("backend")
    weights = np.asarray(bs_recipe["class_multipliers"], dtype=np.float32)
    if weights.shape != (4,) or bs["tree_probability"].shape != (
        *bs["reference"].shape,
        4,
    ):
        raise ValueError("unexpected BS probability/weight shape")
    bs_cnn_weight = 0.0
    if bs_backend == "tree":
        bs_probability = bs["tree_probability"]
    elif bs_backend == "ensemble":
        selected = str(bs_recipe["selected"])
        match = re.fullmatch(r"ensemble_cnn_(0(?:\.\d+)?|1(?:\.0+)?)", selected)
        if not match or bs_cnn_cache_path is None:
            raise ValueError(
                "BS ensemble requires --bs-cnn-cache and ensemble_cnn_* recipe"
            )
        bs_cnn = _bs_cnn_cache(bs_cnn_cache_path, bs_rows)
        if not np.array_equal(
            bs["reference"], bs_cnn["reference"]
        ) or not np.array_equal(bs["valid"], bs_cnn["valid"]):
            raise ValueError(
                "BS tree/CNN reference and valid caches must match exactly"
            )
        bs_cnn_weight = float(match.group(1))
        bs_probability = (
            bs_cnn_weight * bs_cnn["cnn_probability"]
            + (1.0 - bs_cnn_weight) * bs["tree_probability"]
        ).astype(np.float32)
    else:
        raise ValueError(f"unexpected BS recipe backend: {bs_recipe}")
    bs_prediction = (bs_probability * weights).argmax(-1).astype(np.uint8)

    af_all = list(range(len(af_rows)))
    bs_all = list(range(len(bs_rows)))
    af_global = _af_group(af_all, af["reference"], af_prediction, af["valid"])
    bs_global = _bs_group(bs_all, bs["reference"], bs_prediction, bs["valid"])
    if not np.isclose(af_global["f1"], float(af_recipe["F1_af"]), rtol=0.0, atol=1e-15):
        raise ValueError("AF cache reconstruction does not reproduce selected recipe")
    for output_key, recipe_key in (
        ("iou", "IoU_burn"),
        ("miou_severity", "mIoU_sev"),
    ):
        actual = (
            bs_global["burn"][output_key]
            if output_key == "iou"
            else bs_global[output_key]
        )
        if not np.isclose(actual, float(bs_recipe[recipe_key]), rtol=0.0, atol=1e-15):
            raise ValueError(f"BS cache reconstruction does not reproduce {recipe_key}")
    expected_confusion = (
        recipes.get("_postprocess_tuning", {}).get("final", {}).get("confusion")
    )
    confusion_check = None
    if expected_confusion is not None:
        confusion_check = (
            expected_confusion == bs_global["confusion_rows_truth_columns_prediction"]
        )
        if not confusion_check:
            raise ValueError(
                "BS cache reconstruction does not reproduce final confusion"
            )

    af_years = _group_indices(
        af_rows,
        af_meta,
        lambda meta, chip: _year(
            meta["acq_datetime"], field="acq_datetime", chip_id=chip
        ),
    )
    bs_years = _group_indices(
        bs_rows,
        bs_meta,
        lambda meta, chip: _year(meta["date_pre"], field="date_pre", chip_id=chip),
    )

    def cloud_group(meta: dict[str, str], chip_id: str) -> str:
        try:
            value = float(meta["cloud_frac"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"{chip_id}: invalid cloud_frac") from error
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{chip_id}: cloud_frac outside [0,1]")
        if value <= 0.1:
            return "<=0.1"
        if value <= 0.3:
            return "(0.1,0.3]"
        return ">0.3"

    bs_clouds = _group_indices(bs_rows, bs_meta, cloud_group)
    result = {
        "schema_version": 1,
        "scope": "posthoc train-validation subgroup diagnostics; no test data; no tuning",
        "aggregation": "global pixel micro counts within each chip group; never macro-averaged over chips",
        "cache_order_contract": {
            "status": "verified",
            "records": "prepared_records order filtered by task then split=val",
            "evaluate_source": "load_records preserves JSON order; _load_probability appends in that order; np.stack preserves it",
            "af_records": len(af_rows),
            "af_cache_axis0": int(af["reference"].shape[0]),
            "bs_records": len(bs_rows),
            "bs_cache_axis0": int(bs["reference"].shape[0]),
            "selected_recipe_metrics_reproduced": True,
            "bs_final_confusion_reproduced": confusion_check,
        },
        "provenance": {
            "generator": "ops/subgroup_diagnostics.py",
            "inputs": {
                "records": {
                    "path": records_path.as_posix(),
                    "sha256": _sha256(records_path),
                },
                "recipes": {
                    "path": recipes_path.as_posix(),
                    "sha256": _sha256(recipes_path),
                },
                "af_cache": {
                    "path": af_cache_path.as_posix(),
                    "sha256": _sha256(af_cache_path),
                },
                "bs_cache": {
                    "path": bs_cache_path.as_posix(),
                    "sha256": _sha256(bs_cache_path),
                },
                **(
                    {
                        "bs_cnn_cache": {
                            "path": bs_cnn_cache_path.as_posix(),
                            "sha256": _sha256(bs_cnn_cache_path),
                        }
                    }
                    if bs_cnn_cache_path
                    else {}
                ),
            },
        },
        "recipe": {
            "af": {
                "cnn_weight": cnn_weight,
                "tree_weight": tree_weight,
                "threshold": af_threshold,
            },
            "bs": {
                "backend": bs_backend,
                "cnn_weight": bs_cnn_weight,
                "tree_weight": 1.0 - bs_cnn_weight,
                "class_multipliers": weights.astype(float).tolist(),
            },
        },
        "af": {
            "global": af_global,
            "by_acquisition_year": {
                name: _af_group(indices, af["reference"], af_prediction, af["valid"])
                for name, indices in af_years.items()
            },
        },
        "bs": {
            "global": bs_global,
            "by_pre_acquisition_year": {
                name: _bs_group(indices, bs["reference"], bs_prediction, bs["valid"])
                for name, indices in bs_years.items()
            },
            "by_cloud_fraction": {
                "definition": {
                    "<=0.1": "cloud_frac <= 0.1",
                    "(0.1,0.3]": "0.1 < cloud_frac <= 0.3",
                    ">0.3": "cloud_frac > 0.3",
                },
                "groups": {
                    name: _bs_group(
                        indices, bs["reference"], bs_prediction, bs["valid"]
                    )
                    for name, indices in bs_clouds.items()
                },
            },
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post-hoc validation subgroup diagnostics without tuning"
    )
    parser.add_argument("--records", default="artifacts/prepared_records.json")
    parser.add_argument(
        "--af-cache",
        default="artifacts/cnn-v1/af/af_validation_probabilities.npz",
    )
    parser.add_argument(
        "--bs-cache",
        default="artifacts/tree-v1/bs_validation_probabilities.npz",
    )
    parser.add_argument("--bs-cnn-cache")
    parser.add_argument("--af-meta", default="artifacts/train_af_meta.csv")
    parser.add_argument("--bs-meta", default="artifacts/train_bs_meta.csv")
    parser.add_argument(
        "--recipes", default="artifacts/ensemble-v2/selected_recipes.json"
    )
    parser.add_argument("--output", default="-")
    args = parser.parse_args(argv)
    payload = json.dumps(build_report(args), indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        Path(args.output).write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
