"""Validation-only spatial smoothing diagnostic for the frozen V3 BS recipe.

It evaluates a fixed 0.75 tree + 0.25 CNN ensemble and the same ensemble after
per-chip, per-class spatial averaging.  It never tunes a model, multiplier,
or threshold and it never reads test data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops.compare_quality_candidates import (
    WARNING,
    _assert_same,
    _bootstrap,
    _hash_halves,
    _load,
    _metrics,
    _recipe_matches,
)
from ops.bootstrap_bs_comparison import _confusion


V3_CNN_WEIGHT = np.float32(0.25)
V3_MULTIPLIERS = np.asarray((1.0, 2.2, 2.2, 1.3), dtype=np.float32)
WINDOWS = (3, 5)


def _per_chip(
    reference: np.ndarray,
    valid: np.ndarray,
    tree_probability: np.ndarray,
    cnn_probability: np.ndarray,
    *,
    window: int | None,
) -> np.ndarray:
    """Return one BS confusion matrix per validation chip in stable order."""
    matrices = np.zeros((len(reference), 4, 4), dtype=np.int64)
    for index in range(len(reference)):
        probability = (
            np.float32(0.75) * tree_probability[index]
            + V3_CNN_WEIGHT * cnn_probability[index]
        ).astype(np.float32, copy=False)
        if window is not None:
            # size keeps classes independent and never crosses chip boundaries.
            probability = uniform_filter(
                probability, size=(window, window, 1), mode="reflect"
            ).astype(np.float32, copy=False)
        prediction = (probability * V3_MULTIPLIERS).argmax(axis=-1)
        keep = valid[index]
        matrices[index] = _confusion(reference[index][keep], prediction[keep])
    return matrices


def _candidate_report(
    *,
    name: str,
    matrices: np.ndarray,
    baseline: np.ndarray,
    chip_ids: np.ndarray,
    af_f1: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    metrics = _metrics(matrices, af_f1)
    baseline_metrics = _metrics(baseline, af_f1)
    return {
        "name": name,
        "window": None
        if name == "v3_unsmoothed"
        else int(name.removeprefix("uniform_filter_")),
        "spatial_filter": None
        if name == "v3_unsmoothed"
        else {
            "kind": "scipy.ndimage.uniform_filter",
            "mode": "reflect",
            "size_hwc": [
                int(name.removeprefix("uniform_filter_")),
                int(name.removeprefix("uniform_filter_")),
                1,
            ],
        },
        "metrics": metrics,
        "weighted_bs_delta_vs_v3": float(
            metrics["weighted_bs_score"] - baseline_metrics["weighted_bs_score"]
        ),
        "combined_delta_vs_v3": float(
            metrics["combined_score"] - baseline_metrics["combined_score"]
        ),
        "paired_bootstrap_vs_v3": _bootstrap(baseline, matrices, replicates, seed),
        "fixed_hash_halves_vs_v3": _hash_halves(chip_ids, baseline, matrices, af_f1),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    tree = _load(Path(args.tree_cache), "tree", require_ids=False)
    cnn = _load(Path(args.cnn_cache), "cnn", require_ids=True)
    _assert_same(cnn, tree)

    recipes_path = Path(args.recipes)
    recipes = json.loads(recipes_path.read_text(encoding="utf-8"))
    bs_recipe = recipes["bs"]
    if bs_recipe.get("selected") != "ensemble_cnn_0.25":
        raise ValueError("recipes do not identify the frozen V3 0.25 CNN BS ensemble")
    if not np.array_equal(
        np.asarray(bs_recipe.get("class_multipliers"), dtype=np.float32), V3_MULTIPLIERS
    ):
        raise ValueError(
            "recipes do not contain frozen V3 BS multipliers [1, 2.2, 2.2, 1.3]"
        )
    af_f1 = float(recipes["af"]["F1_af"])

    baseline = _per_chip(
        tree["reference"],
        tree["valid"],
        tree["probability"],
        cnn["probability"],
        window=None,
    )
    baseline_metrics = _metrics(baseline, af_f1)
    _recipe_matches(baseline_metrics, bs_recipe)

    candidates = [
        _candidate_report(
            name="v3_unsmoothed",
            matrices=baseline,
            baseline=baseline,
            chip_ids=cnn["chip_ids"],
            af_f1=af_f1,
            replicates=args.replicates,
            seed=args.seed,
        )
    ]
    for window in WINDOWS:
        matrices = _per_chip(
            tree["reference"],
            tree["valid"],
            tree["probability"],
            cnn["probability"],
            window=window,
        )
        candidates.append(
            _candidate_report(
                name=f"uniform_filter_{window}",
                matrices=matrices,
                baseline=baseline,
                chip_ids=cnn["chip_ids"],
                af_f1=af_f1,
                replicates=args.replicates,
                seed=args.seed,
            )
        )
    return {
        "diagnostic_only": True,
        "validation_only": True,
        "warning": "Fixed V3 recipe; no spatial-window, multiplier, threshold, or model selection is authorized from this report. "
        + WARNING,
        "recipe": {
            "tree_weight": 0.75,
            "cnn_weight": 0.25,
            "class_multipliers": V3_MULTIPLIERS.astype(float).tolist(),
            "fixed_AF_F1": af_f1,
        },
        "provenance": {
            "recipes": str(recipes_path.resolve()),
            "tree_cache": {
                "path": str(tree["path"].resolve()),
                "sha256": tree["sha256"],
                "probability_key": tree["key"],
            },
            "cnn_cache": {
                "path": str(cnn["path"].resolve()),
                "sha256": cnn["sha256"],
                "probability_key": cnn["key"],
            },
            "reference_valid_exact": True,
            "chip_id_order_sha256_only": True,
        },
        "baseline_reproduced": {
            "recipe_weighted_bs_score": float(bs_recipe["selection_score"]),
            "recipe_combined_score": float(0.35 * af_f1 + bs_recipe["selection_score"]),
            "exact_metrics_match": True,
        },
        "candidates": candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree-cache", default="artifacts/tree-v1/bs_validation_probabilities.npz"
    )
    parser.add_argument(
        "--cnn-cache", default="artifacts/cnn-v1/bs/validation_probabilities.npz"
    )
    parser.add_argument(
        "--recipes", default="artifacts/ensemble-v3/selected_recipes.json"
    )
    parser.add_argument(
        "--output", default="artifacts/quality-v4/spatial-baseline.json"
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
                "name": item["name"],
                "combined_score": item["metrics"]["combined_score"],
                "combined_delta_vs_v3": item["combined_delta_vs_v3"],
            }
            for item in report["candidates"]
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
