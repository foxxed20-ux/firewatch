"""Train a stronger runtime-compatible LightGBM candidate on the frozen split.

This intentionally reuses the competition feature records and validation logic.
It never discovers or reads test data.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np

from competition.tree import (
    _best_af_threshold,
    _feature_names,
    _full_validation,
    _lgb,
    _sample_train,
    _validation_probabilities,
    load_records,
)


def _deadline_callback(lgb, deadline: float | None):
    """Stop boosting cleanly after the wall-clock budget, keeping the last tree."""

    def callback(env):
        if deadline is not None and time.monotonic() >= deadline:
            raise lgb.callback.EarlyStopException(
                env.iteration, env.evaluation_result_list
            )

    callback.order = 25
    callback.before_iteration = False
    return callback


def train(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    deadline = started + args.time_limit_seconds if args.time_limit_seconds else None
    records = load_records(args.records, args.task)
    train_rows = [row for row in records if row.split == "train"]
    val_rows = [row for row in records if row.split == "val"]
    if not train_rows or not val_rows:
        raise ValueError("both frozen train and validation rows are required")

    x, y, weights, sample = _sample_train(
        train_rows, args.task, args.max_pixels, args.seed
    )
    val_x, val_y, val_weights, val_sample = _sample_train(
        val_rows,
        args.task,
        args.val_pixels,
        args.seed + 1,
        required_classes=False,
    )
    # exponent=1 reproduces the original population-prior correction.  A lower
    # exponent is an explicit training candidate, not a post-hoc score change.
    weights = np.power(weights, args.prior_exponent).astype(np.float32)
    val_weights = np.power(val_weights, args.prior_exponent).astype(np.float32)
    sample["prior_weight_exponent"] = args.prior_exponent
    val_sample["prior_weight_exponent"] = args.prior_exponent

    names = _feature_names(records, x.shape[1])
    params = {
        "objective": "binary" if args.task == "af" else "multiclass",
        "metric": "binary_logloss" if args.task == "af" else "multi_logloss",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": -1,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 1,
        "lambda_l2": args.lambda_l2,
        "verbosity": -1,
        "seed": args.seed,
        "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
        "data_random_seed": args.seed,
        "extra_seed": args.seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": args.num_threads,
    }
    if args.task == "bs":
        params["num_class"] = 4

    lgb = _lgb()
    train_set = lgb.Dataset(
        x, label=y, weight=weights, feature_name=names, free_raw_data=True
    )
    valid_set = lgb.Dataset(
        val_x,
        label=val_y,
        weight=val_weights,
        reference=train_set,
        feature_name=names,
        free_raw_data=True,
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=args.rounds,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(args.early_stopping, verbose=False),
            _deadline_callback(lgb, deadline),
            lgb.log_evaluation(50),
        ],
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output / "model.txt"), num_iteration=booster.best_iteration)
    checkpoint = {
        "format_version": 1,
        "status": "training_complete_validation_pending",
        "candidate": "quality-tree-validation-only",
        "task": args.task,
        "channels": x.shape[1],
        "feature_names": names,
        "threshold": 0.5 if args.task == "af" else None,
        "class_multipliers": [1.0] if args.task == "af" else [1.0] * 4,
        "best_iteration": int(booster.best_iteration),
        "sample": sample,
        "validation_sample": val_sample,
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "seed": args.seed,
        "num_threads": args.num_threads,
        "parameters": params,
        "elapsed_seconds": time.monotonic() - started,
        "warning": "validation only; do not select or tune on test data",
    }
    (output / "metadata.json").write_text(
        json.dumps(checkpoint, indent=2), encoding="utf-8"
    )

    reference, probability, permitted = _validation_probabilities(booster, val_rows)
    reference_array = np.stack(reference).astype(np.uint8, copy=False)
    valid_array = np.stack(permitted).astype(bool, copy=False)
    probability_array = np.stack(probability).astype(np.float32, copy=False)
    if probability_array.ndim != 4:
        raise ValueError("validation probabilities must have shape N,H,W,C")
    cache_path = output / "validation_probabilities.npz"
    temporary_cache = output / "validation_probabilities.npz.tmp"
    with temporary_cache.open("wb") as handle:
        np.savez(
            handle,
            reference=reference_array,
            valid=valid_array,
            lgb_probability=probability_array,
            chip_ids=np.asarray([row.id for row in val_rows]),
        )
    temporary_cache.replace(cache_path)

    if args.task == "af":
        ref = [item[valid] for item, valid in zip(reference, permitted)]
        prob = [
            item[..., 0][valid] for item, valid in zip(probability, permitted)
        ]
        threshold, score = _best_af_threshold(ref, prob)
        metrics, _ = _full_validation(
            reference, probability, permitted, "af", threshold=threshold
        )
        multipliers = [1.0]
    else:
        best_multipliers = np.ones(4, dtype=np.float32)
        metrics, score = _full_validation(
            reference,
            probability,
            permitted,
            "bs",
            multipliers=best_multipliers,
        )
        # Keep this small search comparable with the original trainer.  Final
        # ensemble/postprocessing selection belongs in the existing val-only tools.
        for values in itertools.product((0.7, 1.0, 1.3), repeat=3):
            candidate = np.asarray((1.0,) + values, dtype=np.float32)
            candidate_metrics, candidate_score = _full_validation(
                reference,
                probability,
                permitted,
                "bs",
                multipliers=candidate,
            )
            if candidate_score > score:
                best_multipliers = candidate
                metrics, score = candidate_metrics, candidate_score
        threshold = None
        multipliers = best_multipliers.astype(float).tolist()

    metadata = {
        "format_version": 1,
        "status": "complete",
        "candidate": "quality-tree-validation-only",
        "task": args.task,
        "channels": x.shape[1],
        "feature_names": names,
        "threshold": threshold,
        "class_multipliers": multipliers,
        "best_iteration": int(booster.best_iteration),
        "selection_score": float(score),
        "full_validation": metrics,
        "sample": sample,
        "validation_sample": val_sample,
        "train_records": len(train_rows),
        "val_records": len(val_rows),
        "seed": args.seed,
        "num_threads": args.num_threads,
        "parameters": params,
        "validation_cache": cache_path.name,
        "validation_probability_shape": list(probability_array.shape),
        "elapsed_seconds": time.monotonic() - started,
        "warning": "validation only; do not select or tune on test data",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True, choices=("af", "bs"))
    parser.add_argument("--max-pixels", type=int, default=2_000_000)
    parser.add_argument("--val-pixels", type=int, default=400_000)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--min-data-in-leaf", type=int, default=80)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.9)
    parser.add_argument("--lambda-l2", type=float, default=2.0)
    parser.add_argument("--prior-exponent", type=float, default=1.0)
    parser.add_argument(
        "--time-limit-seconds",
        type=float,
        default=0.0,
        help="wall-clock budget from process setup through boosting; 0 disables",
    )
    args = parser.parse_args(argv)
    for name in (
        "max_pixels",
        "val_pixels",
        "num_threads",
        "rounds",
        "early_stopping",
        "num_leaves",
        "min_data_in_leaf",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("learning_rate", "feature_fraction", "bagging_fraction"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    if (
        args.lambda_l2 < 0
        or not 0 <= args.prior_exponent <= 1
        or args.time_limit_seconds < 0
    ):
        parser.error(
            "lambda-l2/time-limit must be non-negative and prior-exponent in [0, 1]"
        )
    print(json.dumps(train(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
