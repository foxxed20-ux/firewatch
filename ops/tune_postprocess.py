"""Validation-only coordinate tuning of BS probability class multipliers."""

from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from competition.evaluate import _score_confusion

GRID = np.asarray((0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0), dtype=np.float32)


def confusion_from_cache(
    reference, valid, probability, multipliers, chunk_pixels=262_144
):
    """Global 4×4 confusion matrix, keeping memory bounded by ``chunk_pixels``."""
    matrix = np.zeros((4, 4), dtype=np.int64)
    for truth_chip, valid_chip, probability_chip in zip(reference, valid, probability):
        truth = truth_chip.reshape(-1)
        permitted = valid_chip.reshape(-1).astype(bool, copy=False)
        scores = probability_chip.reshape(-1, 4)
        for start in range(0, len(truth), chunk_pixels):
            stop = min(len(truth), start + chunk_pixels)
            keep = permitted[start:stop]
            if not keep.any():
                continue
            predicted = (
                (scores[start:stop] * multipliers).argmax(axis=1).astype(np.uint8)
            )
            codes = truth[start:stop][keep].astype(np.int64) * 4 + predicted[keep]
            matrix += np.bincount(codes, minlength=16).reshape(4, 4)
    return matrix


def score_cache(reference, valid, probability, multipliers, chunk_pixels=262_144):
    matrix = confusion_from_cache(
        reference, valid, probability, multipliers, chunk_pixels
    )
    metrics, score = _score_confusion(matrix)
    return matrix, metrics, float(score)


def tune(reference, valid, probability, initial, chunk_pixels=262_144):
    weights = np.asarray(initial, dtype=np.float32)
    if (
        weights.shape != (4,)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or weights[0] != 1.0
    ):
        raise ValueError(
            "initial class_multipliers must be four positive finite values with weight 0 equal to 1"
        )
    initial_matrix, initial_metrics, initial_score = score_cache(
        reference, valid, probability, weights, chunk_pixels
    )
    best = initial_score
    tried = 0
    log = []
    for pass_index in range(2):
        changed = False
        for class_id in (1, 2, 3):
            starting = float(weights[class_id])
            for candidate in GRID:
                tried += 1
                proposal = weights.copy()
                proposal[class_id] = candidate
                _, metrics, score = score_cache(
                    reference, valid, probability, proposal, chunk_pixels
                )
                if score > best + 1e-12:
                    weights, best, changed = proposal, score, True
                    log.append(
                        {
                            "pass": pass_index + 1,
                            "class_id": class_id,
                            "from": starting,
                            "to": float(candidate),
                            "score": score,
                            "metrics": metrics,
                        }
                    )
                    starting = float(candidate)
        if not changed:
            break
    final_matrix, final_metrics, final_score = score_cache(
        reference, valid, probability, weights, chunk_pixels
    )
    return {
        "initial": {
            "class_multipliers": np.asarray(initial, dtype=np.float32)
            .astype(float)
            .tolist(),
            "metrics": initial_metrics,
            "score": initial_score,
            "confusion": initial_matrix.tolist(),
        },
        "final": {
            "class_multipliers": weights.astype(float).tolist(),
            "metrics": final_metrics,
            "score": final_score,
            "confusion": final_matrix.tolist(),
        },
        "tried_weights": tried,
        "updates": log,
    }


def _recipes(raw):
    data = json.loads(Path(raw).read_text(encoding="utf-8"))
    return data.get("recipes", data) if isinstance(data, dict) else data


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tune BS multipliers on validation cache only"
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--recipe-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--probability-key",
        choices=("tree_probability", "cnn_probability"),
        default="tree_probability",
    )
    parser.add_argument("--chunk-pixels", type=int, default=262_144)
    args = parser.parse_args(argv)
    if args.chunk_pixels <= 0:
        raise ValueError("chunk-pixels must be positive")
    with np.load(args.cache, allow_pickle=False) as archive:
        required = {"reference", "valid", args.probability_key}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"cache lacks {sorted(missing)}")
        raw_reference = np.asarray(archive["reference"])
        valid = np.asarray(archive["valid"], dtype=bool)
        probability = np.asarray(archive[args.probability_key], dtype=np.float32)
    if (
        not np.issubdtype(raw_reference.dtype, np.integer)
        or not np.isin(raw_reference, (0, 1, 2, 3)).all()
    ):
        raise ValueError("validation references must be integer BS 0..3 classes")
    reference = raw_reference.astype(np.uint8, copy=False)
    if reference.shape != valid.shape or probability.shape != (*reference.shape, 4):
        raise ValueError("expected reference/valid [N,H,W] and probability [N,H,W,4]")
    if not np.isfinite(probability).all():
        raise ValueError("probabilities must be finite")
    recipes = _recipes(args.recipe_json)
    if not isinstance(recipes, dict) or "bs" not in recipes:
        raise ValueError("recipe JSON must contain a bs recipe")
    recipe = recipes["bs"]
    if recipe.get("backend") not in {"tree", "cnn"}:
        raise ValueError("only a single tree or cnn BS recipe can be tuned")
    expected_backend = "tree" if args.probability_key == "tree_probability" else "cnn"
    if recipe["backend"] != expected_backend:
        raise ValueError(
            f"{args.probability_key} cannot tune selected {recipe['backend']} recipe"
        )
    report = tune(
        reference, valid, probability, recipe["class_multipliers"], args.chunk_pixels
    )
    output = copy.deepcopy(recipes)
    output["bs"]["class_multipliers"] = report["final"]["class_multipliers"]
    output["bs"]["selection_score"] = report["final"]["score"]
    output["bs"].update(report["final"]["metrics"])
    # Keep AF/BS at the top level so this file is directly consumable by the
    # bundle selector; underscore-prefixed fields are audit metadata.
    result = output
    result["_warning"] = (
        "validation_only: values are selected on this same validation cache, not a test score"
    )
    result["_probability_key"] = args.probability_key
    result["_postprocess_tuning"] = report
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
