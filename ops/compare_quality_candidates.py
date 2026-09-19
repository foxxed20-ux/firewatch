"""Compare validation-only two-model BS candidates against the frozen V3 recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ops.bootstrap_bs_comparison import _confusion, _observed, _score


CNN_WEIGHTS = (0.0, 0.15, 0.25, 0.35, 0.5, 1.0)
MULTIPLIERS = (
    (1.0, 1.0, 1.0, 1.0),
    (1.0, 0.7, 0.7, 0.7),
    (1.0, 1.3, 1.3, 1.3),
    (1.0, 1.6, 1.6, 1.3),
    (1.0, 1.0, 1.0, 0.7),
    (1.0, 1.3, 1.3, 0.7),
    (1.0, 3.0, 3.0, 0.8),
    (1.0, 2.2, 2.2, 1.3),
    (1.0, 2.0, 2.2, 1.3),
    (1.0, 2.4, 2.2, 1.3),
    (1.0, 2.2, 2.0, 1.3),
    (1.0, 2.2, 2.4, 1.3),
    (1.0, 2.2, 2.2, 1.1),
    (1.0, 2.2, 2.2, 1.5),
)
WARNING = (
    "validation-only selection on the same frozen split; bootstrap and hash halves "
    "are post-selection diagnostics and do not remove model-selection bias"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, kind: str, *, require_ids: bool) -> dict[str, Any]:
    keys = ("lgb_probability", "tree_probability") if kind == "tree" else ("cnn_probability",)
    with np.load(path, allow_pickle=False) as archive:
        probability_key = next((key for key in keys if key in archive.files), None)
        if probability_key is None:
            raise ValueError(f"{path} lacks one of {keys}")
        required = {"reference", "valid", probability_key}
        if required - set(archive.files):
            raise ValueError(f"{path} lacks {sorted(required - set(archive.files))}")
        reference = np.asarray(archive["reference"])
        valid = np.asarray(archive["valid"])
        probability = np.asarray(archive[probability_key])
        chip_ids = (
            np.asarray(archive["chip_ids"]).astype(str)
            if "chip_ids" in archive.files
            else None
        )
    if reference.ndim != 3 or not len(reference):
        raise ValueError(f"{path}/reference must be non-empty N,H,W")
    if valid.shape != reference.shape or valid.dtype != np.bool_:
        raise ValueError(f"{path}/valid must be matching bool")
    if probability.shape != (*reference.shape, 4) or probability.dtype != np.float32:
        raise ValueError(f"{path}/{probability_key} must be float32 N,H,W,4")
    if not np.isfinite(probability).all() or np.any(probability < -1e-6) or np.any(probability > 1 + 1e-6):
        raise ValueError(f"{path}/{probability_key} contains invalid probabilities")
    if not np.issubdtype(reference.dtype, np.integer) or not np.isin(reference[valid], (0, 1, 2, 3)).all():
        raise ValueError(f"{path}/reference has invalid scored labels")
    if require_ids and chip_ids is None:
        raise ValueError(f"{path} must contain chip_ids")
    if chip_ids is not None:
        if chip_ids.shape != (len(reference),) or len(set(chip_ids.tolist())) != len(chip_ids):
            raise ValueError(f"{path}/chip_ids must be unique shape N")
    return {
        "path": path,
        "sha256": _sha256(path),
        "key": probability_key,
        "reference": reference.astype(np.uint8, copy=False),
        "valid": valid,
        "probability": probability,
        "chip_ids": chip_ids,
    }


def _assert_same(canonical: dict[str, Any], other: dict[str, Any]) -> None:
    if not np.array_equal(canonical["reference"], other["reference"]):
        raise ValueError(f"reference differs: {other['path']}")
    if not np.array_equal(canonical["valid"], other["valid"]):
        raise ValueError(f"valid differs: {other['path']}")
    if other["chip_ids"] is not None and not np.array_equal(
        canonical["chip_ids"], other["chip_ids"]
    ):
        raise ValueError(f"chip_ids order differs: {other['path']}")


def _per_chip(
    reference: np.ndarray,
    valid: np.ndarray,
    tree_probability: np.ndarray,
    cnn_probability: np.ndarray,
    cnn_weight: float,
    multipliers: tuple[float, float, float, float] | list[float],
) -> np.ndarray:
    matrices = np.zeros((len(reference), 4, 4), dtype=np.int64)
    multiplier = np.asarray(multipliers, dtype=np.float32)
    for index in range(len(reference)):
        combined = (
            (1.0 - cnn_weight) * tree_probability[index]
            + cnn_weight * cnn_probability[index]
        ).astype(np.float32)
        prediction = (combined * multiplier).argmax(axis=-1)
        keep = valid[index]
        matrices[index] = _confusion(reference[index][keep], prediction[keep])
    return matrices


def _metrics(matrices: np.ndarray, fixed_af_f1: float) -> dict[str, Any]:
    result = _observed(matrices.sum(axis=0))
    result["fixed_AF_F1"] = fixed_af_f1
    result["combined_score"] = 0.35 * fixed_af_f1 + result["weighted_bs_score"]
    return result


def _recipe_matches(metrics: dict[str, Any], recipe: dict[str, Any]) -> None:
    pairs = (
        ("IoU_burn", "IoU_burn"),
        ("mIoU_severity", "mIoU_sev"),
        ("IoU_1", "IoU_1"),
        ("IoU_2", "IoU_2"),
        ("IoU_3", "IoU_3"),
        ("weighted_bs_score", "selection_score"),
    )
    for actual, reported in pairs:
        if not np.isclose(metrics[actual], recipe[reported], rtol=0, atol=1e-10):
            raise ValueError(
                f"baseline V3 {actual}={metrics[actual]} differs from recipe {recipe[reported]}"
            )


def _bootstrap(
    baseline: np.ndarray, candidate: np.ndarray, replicates: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(baseline), size=(replicates, len(baseline)))
    baseline_scores = _score(baseline[samples].sum(axis=1))[0]
    candidate_scores = _score(candidate[samples].sum(axis=1))[0]
    difference = candidate_scores - baseline_scores
    observed = _score(candidate.sum(axis=0))[0] - _score(baseline.sum(axis=0))[0]
    return {
        "replicates": replicates,
        "seed": seed,
        "paired": True,
        "observed_weighted_bs_delta": float(observed),
        "percentile_95_interval": np.percentile(difference, (2.5, 97.5)).astype(float).tolist(),
        "fraction_delta_gt_zero": float(np.mean(difference > 0)),
        "warning": WARNING,
    }


def _hash_halves(
    chip_ids: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    fixed_af_f1: float,
) -> dict[str, Any]:
    assignment = np.asarray(
        [int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % 2 for value in chip_ids]
    )
    result = {}
    for half in (0, 1):
        keep = assignment == half
        if not keep.any():
            raise ValueError("hash split produced an empty half")
        base = _metrics(baseline[keep], fixed_af_f1)
        cand = _metrics(candidate[keep], fixed_af_f1)
        result[f"half_{half}"] = {
            "chips": int(keep.sum()),
            "baseline_v3": base,
            "candidate": cand,
            "weighted_bs_delta": cand["weighted_bs_score"] - base["weighted_bs_score"],
        }
    return {
        "method": "sha256(chip_id) integer modulo 2; fixed before metric computation",
        "chip_id_order_sha256": hashlib.sha256("\n".join(chip_ids).encode("utf-8")).hexdigest(),
        "halves": result,
        "warning": WARNING,
    }


def _candidate_spec(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or parts[1] not in {"tree", "cnn"} or not parts[0]:
        raise argparse.ArgumentTypeError("candidate must be NAME=tree|cnn=PATH")
    return parts[0], parts[1], Path(parts[2])


def _pair_spec(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not parts[0]:
        raise argparse.ArgumentTypeError("pair must be NAME=TREE_PATH=CNN_PATH")
    return parts[0], Path(parts[1]), Path(parts[2])


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    baseline_tree = _load(Path(args.baseline_tree), "tree", require_ids=False)
    baseline_cnn = _load(Path(args.baseline_cnn), "cnn", require_ids=True)
    _assert_same(baseline_cnn, baseline_tree)
    recipes_path = Path(args.recipes)
    recipes = json.loads(recipes_path.read_text(encoding="utf-8"))
    bs_recipe = recipes["bs"]
    af_f1 = float(recipes["af"]["F1_af"])
    if bs_recipe.get("selected") != "ensemble_cnn_0.25":
        raise ValueError("recipes do not identify the frozen V3 CNN 0.25 BS baseline")
    baseline_matrices = _per_chip(
        baseline_tree["reference"],
        baseline_tree["valid"],
        baseline_tree["probability"],
        baseline_cnn["probability"],
        0.25,
        bs_recipe["class_multipliers"],
    )
    baseline_metrics = _metrics(baseline_matrices, af_f1)
    _recipe_matches(baseline_metrics, bs_recipe)

    model_pairs = []
    for name, kind, path in args.candidate:
        cache = _load(path, kind, require_ids=True)
        _assert_same(baseline_cnn, cache)
        if kind == "tree":
            model_pairs.append((name, "tree", cache, baseline_cnn))
        else:
            model_pairs.append((name, "cnn", baseline_tree, cache))
    for name, tree_path, cnn_path in args.pair:
        tree_cache = _load(tree_path, "tree", require_ids=True)
        cnn_cache = _load(cnn_path, "cnn", require_ids=True)
        _assert_same(baseline_cnn, tree_cache)
        _assert_same(baseline_cnn, cnn_cache)
        model_pairs.append((name, "pair", tree_cache, cnn_cache))
    if not model_pairs:
        raise ValueError("at least one candidate or explicit pair is required")

    candidates = []
    best = None
    best_matrices = None
    for name, kind, tree_cache, cnn_cache in model_pairs:
        tree = tree_cache["probability"]
        cnn = cnn_cache["probability"]
        sources = {
            "tree": {
                "path": str(tree_cache["path"].resolve()),
                "sha256": tree_cache["sha256"],
                "probability_key": tree_cache["key"],
            },
            "cnn": {
                "path": str(cnn_cache["path"].resolve()),
                "sha256": cnn_cache["sha256"],
                "probability_key": cnn_cache["key"],
            },
        }
        configurations = []
        for cnn_weight in CNN_WEIGHTS:
            for multipliers in MULTIPLIERS:
                matrices = _per_chip(
                    baseline_cnn["reference"],
                    baseline_cnn["valid"],
                    tree,
                    cnn,
                    cnn_weight,
                    multipliers,
                )
                metrics = _metrics(matrices, af_f1)
                item = {
                    "cnn_weight": cnn_weight,
                    "tree_weight": 1.0 - cnn_weight,
                    "class_multipliers": list(multipliers),
                    "metrics": metrics,
                }
                configurations.append(item)
                if best is None or metrics["weighted_bs_score"] > best["metrics"]["weighted_bs_score"]:
                    best = {
                        "candidate": name,
                        "kind": kind,
                        "sources": sources,
                        **item,
                    }
                    best_matrices = matrices
        candidates.append(
            {
                "name": name,
                "kind": kind,
                "sources": sources,
                "configurations": configurations,
            }
        )
    if best is None or best_matrices is None:
        raise RuntimeError("candidate evaluation produced no result")
    return {
        "diagnostic_only": True,
        "warning": WARNING,
        "predetermined_grid": {
            "cnn_weights": list(CNN_WEIGHTS),
            "class_multipliers": [list(value) for value in MULTIPLIERS],
            "two_model_runtime_only": True,
        },
        "provenance": {
            "recipes_sha256": _sha256(recipes_path),
            "baseline_tree_sha256": baseline_tree["sha256"],
            "baseline_cnn_sha256": baseline_cnn["sha256"],
            "canonical_chip_ids_source": str(baseline_cnn["path"]),
            "baseline_tree_chip_ids_present": baseline_tree["chip_ids"] is not None,
            "reference_valid_exact_for_all": True,
            "candidate_chip_ids_exact_for_all": True,
        },
        "baseline_v3": {
            "cnn_weight": 0.25,
            "tree_weight": 0.75,
            "class_multipliers": bs_recipe["class_multipliers"],
            "metrics": baseline_metrics,
        },
        "candidates": candidates,
        "best_candidate": {
            **best,
            "beats_v3_observed": best["metrics"]["weighted_bs_score"]
            > baseline_metrics["weighted_bs_score"],
        },
        "paired_bootstrap_best_minus_v3": _bootstrap(
            baseline_matrices, best_matrices, args.replicates, args.seed
        ),
        "fixed_hash_halves_best_vs_v3": _hash_halves(
            baseline_cnn["chip_ids"], baseline_matrices, best_matrices, af_f1
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-tree", required=True)
    parser.add_argument("--baseline-cnn", required=True)
    parser.add_argument("--recipes", required=True)
    parser.add_argument(
        "--candidate", action="append", default=[], type=_candidate_spec,
        help="repeatable NAME=tree|cnn=PATH",
    )
    parser.add_argument(
        "--pair", action="append", default=[], type=_pair_spec,
        help="repeatable explicit two-model NAME=TREE_PATH=CNN_PATH",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.candidate and not args.pair:
        parser.error("at least one --candidate or --pair is required")
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    report = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
