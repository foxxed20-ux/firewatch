"""Paired chip bootstrap for the frozen V2 LGB and LGB/XGB BS recipes.

Only validation caches are accepted. Recipes are read from their existing
selection reports and are never tuned by this diagnostic.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool | None]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"reference", "valid", probability_key}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        reference = np.asarray(archive["reference"])
        valid = np.asarray(archive["valid"])
        probability = np.asarray(archive[probability_key])
        ids_unique = None
        if "chip_ids" in archive.files:
            chip_ids = np.asarray(archive["chip_ids"])
            if chip_ids.shape != (reference.shape[0],):
                raise ValueError(f"{path}/chip_ids has an incompatible shape")
            ids_unique = len(set(chip_ids.astype(str).tolist())) == len(chip_ids)
    if reference.shape != (45, 512, 512):
        raise ValueError(f"{path}/reference must have shape (45,512,512)")
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
    return reference, valid, probability, ids_unique


def _confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.bincount(
        truth.astype(np.int64) * 4 + prediction.astype(np.int64), minlength=16
    ).reshape(4, 4)


def _per_chip_confusions(
    reference: np.ndarray,
    valid: np.ndarray,
    lgb_probability: np.ndarray,
    xgb_probability: np.ndarray,
    *,
    xgb_weight: float,
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
            (1.0 - xgb_weight) * lgb_probability[index]
            + xgb_weight * xgb_probability[index]
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
    xgb_path = Path(args.xgb_cache)
    comparison_path = Path(args.comparison)
    recipes_path = Path(args.v2_recipes)
    lgb_ref, lgb_valid, lgb_probability, _ = _load_cache(
        lgb_path, "tree_probability"
    )
    xgb_ref, xgb_valid, xgb_probability, xgb_ids_unique = _load_cache(
        xgb_path, "xgb_probability"
    )
    if not np.array_equal(lgb_ref, xgb_ref) or not np.array_equal(
        lgb_valid, xgb_valid
    ):
        raise ValueError("LGB/XGB reference and valid arrays are not exactly equal")
    if xgb_ids_unique is not True:
        raise ValueError("XGB validation cache chip IDs are not unique")

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    recipes = json.loads(recipes_path.read_text(encoding="utf-8"))
    candidate = comparison["best_raw_combined_model_choice"]
    if candidate["name"] != "xgb_weight_0.5":
        raise ValueError(f"unexpected selected comparison candidate: {candidate['name']}")
    xgb_weight = float(candidate["xgb_weight"])
    if float(candidate["lgb_weight"]) != 1.0 - xgb_weight:
        raise ValueError("candidate ensemble weights do not sum to one")
    ensemble_multipliers = np.asarray(
        candidate["final"]["class_multipliers"], dtype=np.float32
    )
    v2_recipe = recipes["bs"]
    if v2_recipe.get("backend") != "tree":
        raise ValueError("V2 BS comparator is not the frozen tree recipe")
    lgb_multipliers = np.asarray(v2_recipe["class_multipliers"], dtype=np.float32)
    if ensemble_multipliers.shape != (4,) or lgb_multipliers.shape != (4,):
        raise ValueError("BS recipes must contain four class multipliers")

    lgb_by_chip, ensemble_by_chip = _per_chip_confusions(
        lgb_ref,
        lgb_valid,
        lgb_probability,
        xgb_probability,
        xgb_weight=xgb_weight,
        lgb_multipliers=lgb_multipliers,
        ensemble_multipliers=ensemble_multipliers,
    )
    lgb_observed = _observed(lgb_by_chip.sum(axis=0))
    ensemble_observed = _observed(ensemble_by_chip.sum(axis=0))
    if lgb_observed["confusion_rows_truth_columns_prediction"] != recipes[
        "_postprocess_tuning"
    ]["final"]["confusion"]:
        raise ValueError("reconstructed V2 LGB confusion does not match its report")
    if ensemble_observed["confusion_rows_truth_columns_prediction"] != candidate[
        "final"
    ]["confusion"]:
        raise ValueError("reconstructed ensemble confusion does not match its report")
    if not np.isclose(
        lgb_observed["weighted_bs_score"],
        float(v2_recipe["selection_score"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("reconstructed V2 LGB score does not match its report")
    if not np.isclose(
        ensemble_observed["weighted_bs_score"],
        float(candidate["final"]["score"]),
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

    return {
        "schema_version": 1,
        "scope": "validation-only paired chip bootstrap; no test data; no tuning",
        "warning": WARNING,
        "n_chips": len(lgb_by_chip),
        "bootstrap_replicates": args.replicates,
        "seed": args.seed,
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
            "xgb_cache_chip_ids_unique": True,
            "reported_confusions_reproduced": True,
            "reported_scores_reproduced": True,
        },
        "recipes": {
            "baseline_v2_lgb": {
                "lgb_weight": 1.0,
                "xgb_weight": 0.0,
                "class_multipliers": lgb_multipliers.astype(float).tolist(),
            },
            "candidate_lgb_xgb": {
                "lgb_weight": 1.0 - xgb_weight,
                "xgb_weight": xgb_weight,
                "class_multipliers": ensemble_multipliers.astype(float).tolist(),
            },
        },
        "observed": {
            "baseline_v2_lgb": lgb_observed,
            "candidate_lgb_xgb": ensemble_observed,
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
            "inputs": {
                "lgb_cache_sha256": _sha256(lgb_path),
                "xgb_cache_sha256": _sha256(xgb_path),
                "comparison_sha256": _sha256(comparison_path),
                "v2_recipes_sha256": _sha256(recipes_path),
            },
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
        "--xgb-cache",
        default="artifacts/xgb-v1/bs/validation_probabilities.npz",
    )
    parser.add_argument(
        "--comparison", default="artifacts/xgb-v1/bs/comparison.json"
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
    report = json.dumps(build_report(args), indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(report)
    else:
        Path(args.output).write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
