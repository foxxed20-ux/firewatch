"""Compare AF D4 CNN TTA against the frozen V3 ensemble on validation chips only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


V3_CNN_WEIGHT = np.float32(0.25)
V3_THRESHOLD = np.float32(0.7525456547737122)
WEIGHTS = tuple(np.float32(value) for value in (0.15, 0.25, 0.35))
WARNING = (
    "Validation-only post-selection comparison: thresholds for the predetermined "
    "weights are optimized on this same held-out split. Bootstrap intervals are "
    "conditional on those thresholds and do not estimate independent generalization."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, *, tta: bool) -> dict[str, np.ndarray | Path | str]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"reference", "valid", "cnn_probability"}
        if not tta:
            required.add("tree_probability")
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        result: dict[str, np.ndarray | Path | str] = {
            "path": path,
            "sha256": _sha256(path),
            "reference": np.asarray(archive["reference"]),
            "valid": np.asarray(archive["valid"]),
            "cnn_probability": np.asarray(archive["cnn_probability"]),
        }
        if "tree_probability" in archive.files:
            result["tree_probability"] = np.asarray(archive["tree_probability"])
        if "chip_ids" in archive.files:
            result["chip_ids"] = np.asarray(archive["chip_ids"]).astype(str)
    reference = result["reference"]
    valid = result["valid"]
    cnn = result["cnn_probability"]
    if (
        not isinstance(reference, np.ndarray)
        or reference.ndim != 3
        or not len(reference)
    ):
        raise ValueError(f"{path}/reference must be nonempty N,H,W")
    if (
        not isinstance(valid, np.ndarray)
        or valid.dtype != np.bool_
        or valid.shape != reference.shape
    ):
        raise ValueError(f"{path}/valid must be matching bool")
    if (
        not isinstance(cnn, np.ndarray)
        or cnn.dtype != np.float32
        or cnn.shape != reference.shape
    ):
        raise ValueError(f"{path}/cnn_probability must be matching float32")
    if not np.isfinite(cnn).all() or np.any(cnn < 0) or np.any(cnn > 1):
        raise ValueError(f"{path}/cnn_probability must be finite probabilities")
    if (
        not np.issubdtype(reference.dtype, np.integer)
        or not np.isin(reference[valid], (0, 1)).all()
    ):
        raise ValueError(f"{path}/reference has invalid scored labels")
    tree = result.get("tree_probability")
    if tree is not None:
        if (
            not isinstance(tree, np.ndarray)
            or tree.dtype != np.float32
            or tree.shape != (*reference.shape, 1)
        ):
            raise ValueError(f"{path}/tree_probability must be float32 N,H,W,1")
        if not np.isfinite(tree).all() or np.any(tree < 0) or np.any(tree > 1):
            raise ValueError(f"{path}/tree_probability must be finite probabilities")
    ids = result.get("chip_ids")
    if ids is not None:
        if (
            not isinstance(ids, np.ndarray)
            or ids.shape != (len(reference),)
            or len(set(ids.tolist())) != len(ids)
        ):
            raise ValueError(f"{path}/chip_ids must be unique N")
    return result


def _best_threshold(truth: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Exact global micro-F1 threshold; ties choose the highest probability."""
    if truth.dtype != np.bool_ or probability.dtype != np.float32:
        raise ValueError("exact threshold expects bool truth and float32 probability")
    positives = int(truth.sum())
    if not len(probability) or positives == 0:
        return float(
            np.nextafter(np.float32(1), np.float32(2))
        ), 1.0 if positives == 0 else 0.0
    order = np.argsort(-probability, kind="stable")
    score = probability[order]
    sorted_truth = truth[order]
    endpoints = np.r_[score[:-1] != score[1:], True]
    positions = np.flatnonzero(endpoints)
    true_positive = np.cumsum(sorted_truth, dtype=np.int64)[positions]
    f1 = 2.0 * true_positive / (positions + 1 + positives)
    winner = int(np.argmax(f1))
    return float(score[positions[winner]]), float(f1[winner])


def _counts(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.count_nonzero(truth & prediction),
            np.count_nonzero(~truth & prediction),
            np.count_nonzero(truth & ~prediction),
        ],
        dtype=np.int64,
    )


def _f1(counts: np.ndarray) -> np.ndarray | float:
    denominator = 2 * counts[..., 0] + counts[..., 1] + counts[..., 2]
    result = np.ones_like(denominator, dtype=np.float64)
    np.divide(2 * counts[..., 0], denominator, out=result, where=denominator != 0)
    return result


def _per_chip_counts(
    reference: np.ndarray, valid: np.ndarray, probability: np.ndarray, threshold: float
) -> np.ndarray:
    values = np.zeros((len(reference), 3), dtype=np.int64)
    for index in range(len(reference)):
        keep = valid[index]
        values[index] = _counts(
            reference[index][keep].astype(bool), probability[index][keep] >= threshold
        )
    return values


def _bootstrap(
    baseline: np.ndarray, candidate: np.ndarray, replicates: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(baseline), size=(replicates, len(baseline)))
    delta = _f1(candidate[samples].sum(axis=1)) - _f1(baseline[samples].sum(axis=1))
    return {
        "unit": "chip (84 chips; conditional)",
        "replicates": replicates,
        "seed": seed,
        "percentile_95_interval": np.percentile(delta, (2.5, 97.5))
        .astype(float)
        .tolist(),
        "fraction_delta_gt_zero": float(np.mean(delta > 0)),
        "warning": WARNING,
    }


def _grid_groups(records: Path, chip_ids: np.ndarray) -> list[np.ndarray]:
    with records.open(encoding="utf-8-sig", newline="") as stream:
        metadata = {str(row["chip_id"]): row for row in csv.DictReader(stream)}
    grouped: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
    for index, chip_id in enumerate(chip_ids):
        row = metadata.get(chip_id)
        if row is None:
            raise ValueError(f"{chip_id}: absent from AF metadata")
        key = tuple(
            row[field] for field in ("epsg", "x_min", "y_min", "x_max", "y_max")
        )
        grouped[key].append(index)
    result = [
        np.asarray(indices, dtype=np.int64) for _, indices in sorted(grouped.items())
    ]
    if len(result) != 17:
        raise ValueError(f"expected 17 AF validation geogrids, got {len(result)}")
    return result


def _grid_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: list[np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    per_group_base = np.stack([baseline[group].sum(axis=0) for group in groups])
    per_group_candidate = np.stack([candidate[group].sum(axis=0) for group in groups])
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(groups), size=(replicates, len(groups)))
    delta = _f1(per_group_candidate[sample].sum(axis=1)) - _f1(
        per_group_base[sample].sum(axis=1)
    )
    return {
        "unit": "exact EPSG/x_min/y_min/x_max/y_max geogrid (17 validation geogrids; conditional)",
        "replicates": replicates,
        "seed": seed,
        "percentile_95_interval": np.percentile(delta, (2.5, 97.5))
        .astype(float)
        .tolist(),
        "fraction_delta_gt_zero": float(np.mean(delta > 0)),
        "warning": WARNING,
    }


def _metrics(
    counts: np.ndarray, baseline_f1: float, bs_fixed: float
) -> dict[str, float | list[int]]:
    f1 = float(_f1(counts.sum(axis=0)))
    combined = 0.35 * f1 + bs_fixed
    return {
        "F1_af": f1,
        "combined_score_with_fixed_bs": combined,
        "delta_F1_vs_v3": f1 - baseline_f1,
        "delta_combined_vs_v3": 0.35 * (f1 - baseline_f1),
        "counts_tp_fp_fn": counts.sum(axis=0).astype(int).tolist(),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _load(Path(args.baseline_cache), tta=False)
    tta = _load(Path(args.tta_cache), tta=True)
    for key in ("reference", "valid"):
        if not np.array_equal(baseline[key], tta[key]):
            raise ValueError(f"baseline and TTA {key} arrays differ")
    chip_ids = tta.get("chip_ids")
    if not isinstance(chip_ids, np.ndarray):
        raise ValueError(
            "TTA cache must carry chip_ids for ordered validation provenance"
        )

    records = json.loads(Path(args.recipes).read_text(encoding="utf-8"))
    af_recipe, bs_recipe = records["af"], records["bs"]
    baseline_f1 = float(af_recipe["F1_af"])
    threshold = float(af_recipe["threshold"])
    if not np.isclose(threshold, float(V3_THRESHOLD), rtol=0.0, atol=1e-7):
        raise ValueError("recipes do not contain frozen V3 AF threshold")
    bs_fixed = float(bs_recipe["selection_score"])

    tree = baseline["tree_probability"]
    cnn = baseline["cnn_probability"]
    tta_cnn = tta["cnn_probability"]
    assert (
        isinstance(tree, np.ndarray)
        and isinstance(cnn, np.ndarray)
        and isinstance(tta_cnn, np.ndarray)
    )
    base_probability = (np.float32(0.75) * tree[..., 0] + V3_CNN_WEIGHT * cnn).astype(
        np.float32
    )
    baseline_counts = _per_chip_counts(
        baseline["reference"], baseline["valid"], base_probability, threshold
    )
    baseline_metrics = _metrics(baseline_counts, baseline_f1, bs_fixed)
    if not np.isclose(baseline_metrics["F1_af"], baseline_f1, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"baseline F1 {baseline_metrics['F1_af']} does not reproduce frozen V3 {baseline_f1}"
        )
    groups = _grid_groups(Path(args.metadata), chip_ids)

    candidate_specs: list[tuple[str, np.float32, float, bool]] = [
        ("tta_cnn_weight_0.25_frozen_threshold", V3_CNN_WEIGHT, threshold, False)
    ]
    candidates: list[dict[str, Any]] = []
    for name, weight, candidate_threshold, optimized in candidate_specs:
        probability = (np.float32(1) - weight) * tree[..., 0] + weight * tta_cnn
        probability = probability.astype(np.float32)
        values = _per_chip_counts(
            baseline["reference"], baseline["valid"], probability, candidate_threshold
        )
        candidates.append(
            {
                "name": name,
                "cnn_weight": float(weight),
                "tree_weight": float(np.float32(1) - weight),
                "threshold": candidate_threshold,
                "threshold_optimized_on_validation": optimized,
                "metrics": _metrics(values, baseline_f1, bs_fixed),
                "chip_bootstrap_vs_v3": _bootstrap(
                    baseline_counts, values, args.replicates, args.seed
                ),
                "geogrid_bootstrap_vs_v3": _grid_bootstrap(
                    baseline_counts, values, groups, args.replicates, args.seed
                ),
            }
        )
    truth = baseline["reference"][baseline["valid"]].astype(bool)
    for weight in WEIGHTS:
        probability = (
            (np.float32(1) - weight) * tree[..., 0] + weight * tta_cnn
        ).astype(np.float32)
        permitted = probability[baseline["valid"]]
        candidate_threshold, _ = _best_threshold(truth, permitted)
        values = _per_chip_counts(
            baseline["reference"], baseline["valid"], probability, candidate_threshold
        )
        candidates.append(
            {
                "name": f"tta_cnn_weight_{float(weight):.2f}_exact_threshold",
                "cnn_weight": float(weight),
                "tree_weight": float(np.float32(1) - weight),
                "threshold": candidate_threshold,
                "threshold_optimized_on_validation": True,
                "metrics": _metrics(values, baseline_f1, bs_fixed),
                "chip_bootstrap_vs_v3": _bootstrap(
                    baseline_counts, values, args.replicates, args.seed
                ),
                "geogrid_bootstrap_vs_v3": _grid_bootstrap(
                    baseline_counts, values, groups, args.replicates, args.seed
                ),
            }
        )
    raw_records = json.loads(Path(args.prepared_records).read_text(encoding="utf-8"))
    prepared_records = (
        raw_records["records"] if isinstance(raw_records, dict) else raw_records
    )
    record_ids = [
        str(row["id"])
        for row in prepared_records
        if str(row["task"]).lower() == "af" and str(row["split"]) == "val"
    ]
    if chip_ids.tolist() != record_ids:
        raise ValueError("TTA chip_ids differ from prepared AF validation record order")
    return {
        "diagnostic_only": True,
        "validation_only": True,
        "warning": WARNING,
        "provenance": {
            "baseline_cache": {
                "path": str(Path(args.baseline_cache).resolve()),
                "sha256": baseline["sha256"],
            },
            "tta_cache": {
                "path": str(Path(args.tta_cache).resolve()),
                "sha256": tta["sha256"],
            },
            "reference_valid_exact": True,
            "tta_chip_ids_verified_against_records": True,
            "metadata_geogrids": len(groups),
        },
        "frozen_v3": {
            "cnn_weight": 0.25,
            "tree_weight": 0.75,
            "threshold": threshold,
            "F1_af": baseline_f1,
            "fixed_bs_score": bs_fixed,
            "combined_score": 0.35 * baseline_f1 + bs_fixed,
        },
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-cache",
        default="artifacts/cnn-v1/af/af_validation_probabilities.npz",
    )
    parser.add_argument(
        "--tta-cache", default="artifacts/quality-v4/tta-baseline/af.npz"
    )
    parser.add_argument(
        "--recipes", default="artifacts/ensemble-v3/selected_recipes.json"
    )
    parser.add_argument("--prepared-records", default="artifacts/prepared_records.json")
    parser.add_argument("--metadata", default="artifacts/train_af_meta.csv")
    parser.add_argument(
        "--output", default="artifacts/quality-v4/tta-af-comparison.json"
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    summary = {
        "output": str(output),
        "candidates": [
            {
                "name": candidate["name"],
                "F1_af": candidate["metrics"]["F1_af"],
                "delta_combined": candidate["metrics"]["delta_combined_vs_v3"],
            }
            for candidate in report["candidates"]
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
