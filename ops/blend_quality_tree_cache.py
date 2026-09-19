"""Build a fixed equal-weight validation cache from two tree probability caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_cache(path: Path, *, require_ids: bool) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        key = next(
            (name for name in ("lgb_probability", "tree_probability") if name in archive.files),
            None,
        )
        if key is None or not {"reference", "valid"}.issubset(archive.files):
            raise ValueError(f"{path} lacks the tree validation-cache contract")
        reference = np.asarray(archive["reference"])
        valid = np.asarray(archive["valid"])
        probability = np.asarray(archive[key])
        chip_ids = (
            np.asarray(archive["chip_ids"]).astype(str)
            if "chip_ids" in archive.files
            else None
        )
    if reference.ndim != 3 or valid.shape != reference.shape or valid.dtype != np.bool_:
        raise ValueError(f"{path} has invalid reference/valid arrays")
    if probability.shape != (*reference.shape, 4) or probability.dtype != np.float32:
        raise ValueError(f"{path}/{key} must be float32 N,H,W,4")
    if not np.isfinite(probability).all():
        raise ValueError(f"{path}/{key} contains non-finite values")
    if require_ids and chip_ids is None:
        raise ValueError(f"{path} must contain chip_ids")
    if chip_ids is not None and (
        chip_ids.shape != (len(reference),)
        or len(set(chip_ids.tolist())) != len(chip_ids)
    ):
        raise ValueError(f"{path}/chip_ids must be unique shape N")
    return {
        "path": path,
        "sha256": _sha256(path),
        "probability_key": key,
        "reference": reference,
        "valid": valid,
        "probability": probability,
        "chip_ids": chip_ids,
    }


def _canonical_ids(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {"reference", "valid", "chip_ids"}
        if required - set(archive.files):
            raise ValueError(f"{path} lacks {sorted(required - set(archive.files))}")
        reference = np.asarray(archive["reference"])
        valid = np.asarray(archive["valid"])
        chip_ids = np.asarray(archive["chip_ids"]).astype(str)
    if valid.shape != reference.shape or valid.dtype != np.bool_:
        raise ValueError(f"{path} has invalid reference/valid arrays")
    if chip_ids.shape != (len(reference),) or len(set(chip_ids.tolist())) != len(chip_ids):
        raise ValueError(f"{path}/chip_ids must be unique shape N")
    return {
        "path": path,
        "sha256": _sha256(path),
        "reference": reference,
        "valid": valid,
        "chip_ids": chip_ids,
    }


def _same_contract(left: dict, right: dict, label: str) -> None:
    if not np.array_equal(left["reference"], right["reference"]):
        raise ValueError(f"{label} reference differs")
    if not np.array_equal(left["valid"], right["valid"]):
        raise ValueError(f"{label} valid differs")


def build(args: argparse.Namespace) -> dict:
    original = _tree_cache(Path(args.original_tree), require_ids=False)
    quality = _tree_cache(Path(args.quality_tree), require_ids=True)
    canonical = _canonical_ids(Path(args.canonical_ids_cache))
    _same_contract(canonical, original, "original tree/canonical")
    _same_contract(canonical, quality, "quality tree/canonical")
    if original["chip_ids"] is not None and not np.array_equal(
        original["chip_ids"], canonical["chip_ids"]
    ):
        raise ValueError("original tree chip_ids order differs from canonical")
    if not np.array_equal(quality["chip_ids"], canonical["chip_ids"]):
        raise ValueError("quality tree chip_ids order differs from canonical")

    # Reuse the original probability allocation to keep peak memory bounded.
    # np.add and np.multiply retain float32 because all operands and the scalar are float32.
    probability = original["probability"]
    np.add(probability, quality["probability"], out=probability)
    np.multiply(probability, np.float32(0.5), out=probability)
    if not np.isfinite(probability).all():
        raise ValueError("mixed probability contains non-finite values")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            reference=canonical["reference"].astype(np.uint8, copy=False),
            valid=canonical["valid"],
            lgb_probability=probability,
            chip_ids=canonical["chip_ids"],
        )
    temporary.replace(output)
    report = {
        "validation_only": True,
        "operation": "float32(original_tree + quality_tree) * float32(0.5)",
        "weights": {"original_tree": 0.5, "quality_tree": 0.5},
        "inputs_in_arithmetic_order": [
            {
                "role": "original_tree",
                "path": str(original["path"].resolve()),
                "sha256": original["sha256"],
                "probability_key": original["probability_key"],
                "probability_dtype": str(original["probability"].dtype),
            },
            {
                "role": "quality_tree",
                "path": str(quality["path"].resolve()),
                "sha256": quality["sha256"],
                "probability_key": quality["probability_key"],
                "probability_dtype": str(quality["probability"].dtype),
            },
        ],
        "canonical_chip_ids": {
            "path": str(canonical["path"].resolve()),
            "sha256": canonical["sha256"],
            "order_sha256": hashlib.sha256(
                "\n".join(canonical["chip_ids"]).encode("utf-8")
            ).hexdigest(),
        },
        "contract": {
            "reference_valid_exact_for_all_inputs": True,
            "chip_ids_exact_for_quality_tree": True,
            "original_tree_ids_source": "canonical_ids_cache after exact reference/valid match",
            "shape": list(probability.shape),
            "dtype": str(probability.dtype),
            "probability_key": "lgb_probability",
        },
        "output": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "warning": "validation cache only; no test data and no calibration before mixing",
    }
    sidecar = Path(args.sidecar) if args.sidecar else output.with_suffix(".json")
    sidecar.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-tree", required=True)
    parser.add_argument("--quality-tree", required=True)
    parser.add_argument("--canonical-ids-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sidecar")
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
