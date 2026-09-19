"""Validation-only LGB/XGB BS probability ensemble comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ops.tune_postprocess import score_cache, tune

INITIAL = np.asarray((1.0, 3.0, 3.0, 0.8), dtype=np.float32)
XGB_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _load(path, probability_key):
    with np.load(path, allow_pickle=False) as archive:
        required = {"reference", "valid", probability_key}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        reference = np.asarray(archive["reference"])
        valid = np.asarray(archive["valid"])
        probability = np.asarray(archive[probability_key])
    if probability.dtype != np.float32 or not np.isfinite(probability).all():
        raise ValueError(f"{path}/{probability_key} must be finite float32")
    if (
        not np.issubdtype(reference.dtype, np.integer)
        or not np.isin(reference, (0, 1, 2, 3)).all()
    ):
        raise ValueError(f"{path}/reference must be integer BS 0..3")
    if valid.dtype != np.bool_:
        raise ValueError(f"{path}/valid must be bool")
    if probability.shape != (*reference.shape, 4) or valid.shape != reference.shape:
        raise ValueError(f"{path} has incompatible cache shapes")
    return reference.astype(np.uint8, copy=False), valid, probability


def compare(
    reference,
    valid,
    lgb_probability,
    secondary_probability,
    chunk_pixels=262_144,
    secondary_kind="xgb",
):
    if secondary_kind not in {"xgb", "cnn"}:
        raise ValueError("secondary_kind must be xgb or cnn")
    result = {}
    for secondary_weight in XGB_WEIGHTS:
        combined = (
            (1.0 - secondary_weight) * lgb_probability
            + secondary_weight * secondary_probability
        ).astype(np.float32)
        if secondary_weight == 0.0:
            _, metrics, score = score_cache(
                reference, valid, combined, INITIAL, chunk_pixels
            )
            report = {
                "initial": {
                    "class_multipliers": INITIAL.astype(float).tolist(),
                    "metrics": metrics,
                    "score": score,
                },
                "final": {
                    "class_multipliers": INITIAL.astype(float).tolist(),
                    "metrics": metrics,
                    "score": score,
                },
                "tried_weights": 0,
                "updates": [],
            }
        else:
            report = tune(reference, valid, combined, INITIAL, chunk_pixels)
        result[f"{secondary_kind}_weight_{secondary_weight:g}"] = {
            f"{secondary_kind}_weight": secondary_weight,
            "lgb_weight": 1.0 - secondary_weight,
            "final": report["final"],
            "initial": report["initial"],
            "tried_weights": report["tried_weights"],
            "updates": report["updates"],
        }
    name, best = max(result.items(), key=lambda item: item[1]["final"]["score"])
    return {
        "candidates": result,
        "best_raw_combined_model_choice": {"name": name, **best},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnostic-only validation comparison of LGB and a secondary BS model"
    )
    parser.add_argument("--lgb-cache", required=True)
    parser.add_argument(
        "--secondary-cache",
        "--xgb-cache",
        dest="secondary_cache",
        required=True,
        help="secondary cache; --xgb-cache is retained as a compatibility alias",
    )
    parser.add_argument("--secondary-kind", choices=("xgb", "cnn"), default="xgb")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-pixels", type=int, default=262_144)
    args = parser.parse_args(argv)
    if args.chunk_pixels <= 0:
        raise ValueError("chunk-pixels must be positive")
    lgb_ref, lgb_valid, lgb_probability = _load(args.lgb_cache, "tree_probability")
    probability_key = f"{args.secondary_kind}_probability"
    secondary_ref, secondary_valid, secondary_probability = _load(
        args.secondary_cache, probability_key
    )
    if not np.array_equal(lgb_ref, secondary_ref) or not np.array_equal(
        lgb_valid, secondary_valid
    ):
        raise ValueError(
            "LGB and secondary cache reference/valid arrays must be exactly equal"
        )
    if lgb_probability.shape != secondary_probability.shape:
        raise ValueError("LGB and secondary probability shapes must be identical")
    report = compare(
        lgb_ref,
        lgb_valid,
        lgb_probability,
        secondary_probability,
        chunk_pixels=args.chunk_pixels,
        secondary_kind=args.secondary_kind,
    )
    report.update(
        {
            "diagnostic_only": True,
            "warning": "validation_only: this comparison must not use test data or change runtime",
            "initial_multipliers": INITIAL.astype(float).tolist(),
            "secondary_kind": args.secondary_kind,
        }
    )
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
