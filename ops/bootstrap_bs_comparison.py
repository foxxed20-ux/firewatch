"""Paired chip bootstrap for frozen V2 LGB and LGB/secondary BS recipes.

Only validation caches are accepted. The V2 baseline comes from its selection
report; the frozen secondary weight and multipliers come from CLI arguments.
This diagnostic never tunes either recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


WARNING = (
    "conditional on selected recipes; does not include model-selection bias; "
    "approximate independent event chips"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cache(
    path: Path, probability_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"reference", "valid", probability_key}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        reference = np.asarray(archive["reference"])
        valid = np.asarray(archive["valid"])
        probability = np.asarray(archive[probability_key])
        chip_ids = None
        if "chip_ids" in archive.files:
            chip_ids = np.asarray(archive["chip_ids"])
            if chip_ids.shape != (reference.shape[0],):
                raise ValueError(f"{path}/chip_ids has an incompatible shape")
            if len(set(chip_ids.astype(str).tolist())) != len(chip_ids):
                raise ValueError(f"{path}/chip_ids are not unique")
    if reference.ndim != 3 or reference.shape[0] == 0:
        raise ValueError(f"{path}/reference must be a non-empty [N,H,W] array")
    if valid.shape != reference.shape or valid.dtype != np.bool_:
        raise ValueError(f"{path}/valid must be a matching bool array")
    if probability.shape != (*reference.shape, 4):
        raise ValueError(f"{path}/{probability_key} has an incompatible shape")
    if probability.dtype != np.float32 or not np.isfinite(probability).all():
        raise ValueError(f"{path}/{probability_key} must be finite float32")
    if not np.issubdtype(reference.dtype, np.integer):
        raise ValueError(f"{path}/reference must be integer")
    if not np.isin(reference[valid], (0, 1, 2, 3)).all():
        raise ValueError(f"{path}/reference has invalid scored labels")
    return reference, valid, probability, chip_ids


def _confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.bincount(
        truth.astype(np.int64) * 4 + prediction.astype(np.int64), minlength=16
    ).reshape(4, 4)


def _per_chip_confusions(
    reference: np.ndarray,
    valid: np.ndarray,
    lgb_probability: np.ndarray,
    secondary_probability: np.ndarray,
    *,
    secondary_weight: float,
    lgb_multipliers: np.ndarray,
    ensemble_multipliers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lgb_matrices = np.zeros((len(reference), 4, 4), dtype=np.int64)
    ensemble_matrices = np.zeros_like(lgb_matrices)
    for index in range(len(reference)):
        keep = valid[index]
        truth = reference[index][keep]
        lgb_prediction = (
            lgb_probability[index] * lgb_multipliers
        ).argmax(-1)[keep]
        combined = (
            (1.0 - secondary_weight) * lgb_probability[index]
            + secondary_weight * secondary_probability[index]
        ).astype(np.float32)
        ensemble_prediction = (combined * ensemble_multipliers).argmax(-1)[keep]
        lgb_matrices[index] = _confusion(truth, lgb_prediction)
        ensemble_matrices[index] = _confusion(truth, ensemble_prediction)
    return lgb_matrices, ensemble_matrices


def _score(confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted BS score, burn IoU, severity mIoU and class IoUs."""
    burn_tp = confusion[..., 1:, 1:].sum(axis=(-2, -1))
    burn_fp = confusion[..., 0, 1:].sum(axis=-1)
    burn_fn = confusion[..., 1:, 0].sum(axis=-1)
    burn_union = burn_tp + burn_fp + burn_fn
    burn = np.ones_like(burn_union, dtype=np.float64)
    np.divide(burn_tp, burn_union, out=burn, where=burn_union != 0)

    severity = []
    for class_id in (1, 2, 3):
        true_positive = confusion[..., class_id, class_id]
        union = (
            confusion[..., class_id, :].sum(axis=-1)
            + confusion[..., :, class_id].sum(axis=-1)
            - true_positive
        )
        class_iou = np.ones_like(union, dtype=np.float64)
        np.divide(true_positive, union, out=class_iou, where=union != 0)
        severity.append(class_iou)
    severity_iou = np.stack(severity, axis=-1)
    severity_mean = (
        severity_iou[..., 0] + severity_iou[..., 1] + severity_iou[..., 2]
    ) / 3
    weighted_score = 0.35 * burn + 0.30 * severity_mean
    return weighted_score, burn, severity_mean, severity_iou


def _observed(confusion: np.ndarray) -> dict[str, Any]:
    burn_tp = int(confusion[1:, 1:].sum())
    burn_fp = int(confusion[0, 1:].sum())
    burn_fn = int(confusion[1:, 0].sum())
    burn_union = burn_tp + burn_fp + burn_fn
    burn = 1.0 if burn_union == 0 else float(burn_tp / burn_union)
    severity = []
    for class_id in (1, 2, 3):
        true_positive = int(confusion[class_id, class_id])
        union = int(
            confusion[class_id, :].sum()
            + confusion[:, class_id].sum()
            - true_positive
        )
        severity.append(1.0 if union == 0 else float(true_positive / union))
    severity_mean = float(sum(severity) / 3)
    score = 0.35 * burn + 0.30 * severity_mean
    return {
        "confusion_rows_truth_columns_prediction": confusion.astype(int).tolist(),
        "IoU_burn": burn,
        "mIoU_severity": severity_mean,
        "IoU_1": severity[0],
        "IoU_2": severity[1],
        "IoU_3": severity[2],
        "weighted_bs_score": score,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    lgb_path = Path(args.lgb_cache)
    secondary_path = Path(args.secondary_cache)
    comparison_path = Path(args.comparison)
    recipes_path = Path(args.v2_recipes)
    secondary_kind = str(args.secondary_kind)
    probability_key = f"{secondary_kind}_probability"
    lgb_ref, lgb_valid, lgb_probability, lgb_ids = _load_cache(
        lgb_path, "tree_probability"
    )
    secondary_ref, secondary_valid, secondary_probability, secondary_ids = (
        _load_cache(secondary_path, probability_key)
    )
    if not np.array_equal(lgb_ref, secondary_ref) or not np.array_equal(
        lgb_valid, secondary_valid
    ):
        raise ValueError("LGB/secondary reference and valid arrays are not exactly equal")
    ids_order_equal = None
    if lgb_ids is not None and secondary_ids is not None:
        if not np.array_equal(lgb_ids.astype(str), secondary_ids.astype(str)):
            raise ValueError("LGB/secondary cache chip ID order differs")
        ids_order_equal = True

    recipes = json.loads(recipes_path.read_text(encoding="utf-8"))
    secondary_weight = float(args.secondary_weight)
    ensemble_multipliers = np.asarray(args.secondary_multipliers, dtype=np.float32)
    v2_recipe = recipes["bs"]
    if v2_recipe.get("backend") != "tree":
        raise ValueError("V2 BS comparator is not the frozen tree recipe")
    lgb_multipliers = np.asarray(v2_recipe["class_multipliers"], dtype=np.float32)
    if ensemble_multipliers.shape != (4,) or lgb_multipliers.shape != (4,):
        raise ValueError("BS recipes must contain four class multipliers")

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    reported_candidate = comparison["best_raw_combined_model_choice"]
    reported_multipliers = np.asarray(
        reported_candidate["final"]["class_multipliers"], dtype=np.float32
    )
    expected_name = f"{secondary_kind}_weight_{secondary_weight:g}"
    if reported_candidate.get("name") != expected_name:
        raise ValueError(
            f"comparison selected {reported_candidate.get('name')!r}, expected {expected_name!r}"
        )
    if (
        float(reported_candidate[f"{secondary_kind}_weight"]) != secondary_weight
        or float(reported_candidate["lgb_weight"]) != 1.0 - secondary_weight
        or not np.array_equal(reported_multipliers, ensemble_multipliers)
    ):
        raise ValueError("comparison weights/multipliers differ from CLI recipe")

    lgb_by_chip, ensemble_by_chip = _per_chip_confusions(
        lgb_ref,
        lgb_valid,
        lgb_probability,
        secondary_probability,
        secondary_weight=secondary_weight,
        lgb_multipliers=lgb_multipliers,
        ensemble_multipliers=ensemble_multipliers,
    )
    lgb_observed = _observed(lgb_by_chip.sum(axis=0))
    ensemble_observed = _observed(ensemble_by_chip.sum(axis=0))
    if lgb_observed["confusion_rows_truth_columns_prediction"] != recipes[
        "_postprocess_tuning"
    ]["final"]["confusion"]:
        raise ValueError("reconstructed V2 LGB confusion does not match its report")
    if not np.isclose(
        lgb_observed["weighted_bs_score"],
        float(v2_recipe["selection_score"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("reconstructed V2 LGB score does not match its report")
    if ensemble_observed["confusion_rows_truth_columns_prediction"] != (
        reported_candidate["final"]["confusion"]
    ):
        raise ValueError("reconstructed ensemble confusion does not match its report")
    if not np.isclose(
        ensemble_observed["weighted_bs_score"],
        float(reported_candidate["final"]["score"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("reconstructed ensemble score does not match its report")

    rng = np.random.default_rng(args.seed)
    samples = rng.integers(
        0, len(lgb_by_chip), size=(args.replicates, len(lgb_by_chip))
    )
    lgb_bootstrap = lgb_by_chip[samples].sum(axis=1)
    ensemble_bootstrap = ensemble_by_chip[samples].sum(axis=1)
    lgb_scores = _score(lgb_bootstrap)[0]
    ensemble_scores = _score(ensemble_bootstrap)[0]
    difference = ensemble_scores - lgb_scores
    interval = np.percentile(difference, (2.5, 97.5), method="linear")
    observed_delta = (
        ensemble_observed["weighted_bs_score"] - lgb_observed["weighted_bs_score"]
    )

    candidate_key = f"candidate_lgb_{secondary_kind}"
    secondary_weight_key = f"{secondary_kind}_weight"
    provenance_inputs = {
        "lgb_cache_sha256": _sha256(lgb_path),
        f"{secondary_kind}_cache_sha256": _sha256(secondary_path),
        "v2_recipes_sha256": _sha256(recipes_path),
    }
    provenance_inputs["comparison_sha256"] = _sha256(comparison_path)

    return {
        "schema_version": 1,
        "scope": "validation-only paired chip bootstrap; no test data; no tuning",
        "warning": WARNING,
        "n_chips": len(lgb_by_chip),
        "bootstrap_replicates": args.replicates,
        "seed": args.seed,
        "secondary_kind": secondary_kind,
        "resampling": {
            "unit": "BS validation chip",
            "draws_per_replicate": len(lgb_by_chip),
            "replacement": True,
            "pairing": "same sampled chip indices for both frozen recipes",
            "aggregation": "sum sampled chip confusion matrices, then compute global micro score",
            "interval": "2.5th and 97.5th percentiles of paired score differences",
        },
        "cache_contract": {
            "reference_exactly_equal": True,
            "valid_exactly_equal": True,
            "probability_shape": list(lgb_probability.shape),
            "lgb_cache_has_chip_ids": lgb_ids is not None,
            f"{secondary_kind}_cache_has_chip_ids": secondary_ids is not None,
            f"{secondary_kind}_cache_chip_ids_unique": (
                True if secondary_ids is not None else None
            ),
            "chip_id_order_exactly_equal": ids_order_equal,
            "baseline_reported_confusion_reproduced": True,
            "baseline_reported_score_reproduced": True,
            "reported_confusions_reproduced": True,
            "reported_scores_reproduced": True,
            "candidate_reported_confusion_reproduced": True,
            "candidate_reported_score_reproduced": True,
        },
        "recipes": {
            "baseline_v2_lgb": {
                "lgb_weight": 1.0,
                secondary_weight_key: 0.0,
                "class_multipliers": lgb_multipliers.astype(float).tolist(),
            },
            candidate_key: {
                "lgb_weight": 1.0 - secondary_weight,
                secondary_weight_key: secondary_weight,
                "class_multipliers": ensemble_multipliers.astype(float).tolist(),
            },
        },
        "observed": {
            "baseline_v2_lgb": lgb_observed,
            candidate_key: ensemble_observed,
            "paired_weighted_bs_score_delta_candidate_minus_baseline": float(
                observed_delta
            ),
        },
        "bootstrap": {
            "paired_delta_mean": float(difference.mean()),
            "paired_delta_median": float(np.median(difference)),
            "paired_delta_sample_std": float(difference.std(ddof=1)),
            "percentile_95_interval": [float(interval[0]), float(interval[1])],
        },
        "provenance": {
            "generator": "ops/bootstrap_bs_comparison.py",
            "inputs": provenance_inputs,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Paired chip bootstrap for frozen BS validation recipes"
    )
    parser.add_argument(
        "--lgb-cache",
        default="artifacts/tree-v1/bs_validation_probabilities.npz",
    )
    parser.add_argument(
        "--secondary-cache",
        "--xgb-cache",
        dest="secondary_cache",
        default="artifacts/xgb-v1/bs/validation_probabilities.npz",
    )
    parser.add_argument(
        "--secondary-kind", choices=("cnn", "xgb"), default="xgb"
    )
    parser.add_argument("--secondary-weight", type=float, default=0.5)
    parser.add_argument(
        "--secondary-multipliers",
        type=float,
        nargs=4,
        default=(1.0, 3.0, 3.0, 1.0),
        metavar=("CLASS0", "CLASS1", "CLASS2", "CLASS3"),
    )
    parser.add_argument(
        "--comparison",
        default="artifacts/xgb-v1/bs/comparison.json",
        help="selection comparison report for the frozen secondary recipe",
    )
    parser.add_argument(
        "--v2-recipes", default="artifacts/ensemble-v2/selected_recipes.json"
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="-")
    args = parser.parse_args(argv)
    if args.replicates <= 0:
        raise ValueError("replicates must be positive")
    if not np.isfinite(args.secondary_weight) or not 0 <= args.secondary_weight <= 1:
        raise ValueError("secondary-weight must be finite and within [0,1]")
    if not all(np.isfinite(value) and value > 0 for value in args.secondary_multipliers):
        raise ValueError("secondary-multipliers must be four positive finite values")
    report = json.dumps(build_report(args), indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(report)
    else:
        Path(args.output).write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
