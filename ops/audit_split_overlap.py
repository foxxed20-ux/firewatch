"""Audit positive-area train/validation grid overlap without exposing locations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


BOUNDS_FIELDS = ("x_min", "y_min", "x_max", "y_max")


def _file_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return {"name": path.name, "size_bytes": size, "sha256": digest.hexdigest()}


def _split(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = raw.get("split", raw.get("assignments", raw)) if isinstance(raw, dict) else raw
    if not isinstance(mapping, dict):
        raise ValueError("split must be a mapping or contain a split/assignments mapping")
    result: dict[str, str] = {}
    for key, value in mapping.items():
        partition = value.get("split") if isinstance(value, dict) else value
        if partition not in {"train", "val"}:
            raise ValueError("every split assignment must be train or val")
        result[str(key)] = str(partition)
    return result


def _epsg(value: str) -> int:
    normalized = value.strip().upper()
    if normalized.startswith("EPSG:"):
        normalized = normalized[5:]
    number = float(normalized)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError("EPSG must be a finite integer code")
    return int(number)


def _task_grids(
    metadata_path: Path, assignments: dict[str, str]
) -> tuple[dict[str, set[tuple[float | int, ...]]], set[str], int]:
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"chip_id", "epsg", *BOUNDS_FIELDS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{metadata_path}: missing chip/grid fields")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{metadata_path}: metadata is empty")

    grids: dict[str, set[tuple[float | int, ...]]] = {
        "train": set(),
        "val": set(),
    }
    chip_ids: set[str] = set()
    for row in rows:
        chip_id = str(row["chip_id"])
        if chip_id in chip_ids:
            raise ValueError(f"{metadata_path}: duplicate chip_id")
        chip_ids.add(chip_id)
        partition = assignments.get(chip_id)
        if partition not in grids:
            raise ValueError(f"{metadata_path}: missing split assignment")
        bounds = tuple(float(row[field]) for field in BOUNDS_FIELDS)
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError(f"{metadata_path}: non-finite bounds")
        x_min, y_min, x_max, y_max = bounds
        if not (x_min < x_max and y_min < y_max):
            raise ValueError(f"{metadata_path}: non-positive grid area")
        grids[partition].add((_epsg(row["epsg"]), *bounds))
    return grids, chip_ids, len(rows)


def _audit_task(
    metadata_path: Path, assignments: dict[str, str]
) -> tuple[dict[str, Any], set[str]]:
    grids, chip_ids, chips = _task_grids(metadata_path, assignments)
    train = grids["train"]
    validation = grids["val"]
    exact = train & validation
    positive_pairs = 0
    affected_train: set[tuple[float | int, ...]] = set()
    affected_validation: set[tuple[float | int, ...]] = set()
    for val_grid in validation:
        epsg, val_x_min, val_y_min, val_x_max, val_y_max = val_grid
        for train_grid in train:
            if train_grid[0] != epsg:
                continue
            _, train_x_min, train_y_min, train_x_max, train_y_max = train_grid
            overlap_width = min(val_x_max, train_x_max) - max(
                val_x_min, train_x_min
            )
            overlap_height = min(val_y_max, train_y_max) - max(
                val_y_min, train_y_min
            )
            if overlap_width > 0 and overlap_height > 0:
                positive_pairs += 1
                affected_train.add(train_grid)
                affected_validation.add(val_grid)
    return (
        {
            "chips": chips,
            "train_unique_grids": len(train),
            "validation_unique_grids": len(validation),
            "exact_cross_split_unique_grids": len(exact),
            "positive_area_cross_split_unique_grid_pairs": positive_pairs,
            "affected_train_unique_grids": len(affected_train),
            "affected_validation_unique_grids": len(affected_validation),
            "passes_no_positive_area_overlap": positive_pairs == 0,
        },
        chip_ids,
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    af_path = Path(args.af_meta)
    bs_path = Path(args.bs_meta)
    split_path = Path(args.split)
    assignments = _split(split_path)
    af, af_ids = _audit_task(af_path, assignments)
    bs, bs_ids = _audit_task(bs_path, assignments)
    if af_ids & bs_ids:
        raise ValueError("AF and BS metadata contain duplicate chip IDs")
    if af_ids | bs_ids != set(assignments):
        raise ValueError("metadata and split do not have identical chip coverage")
    return {
        "schema_version": 1,
        "scope": {
            "partitions": "train versus validation",
            "geometry": "axis-aligned metadata bounds within the same EPSG",
            "positive_area_rule": "overlap width > 0 and overlap height > 0",
            "edge_or_corner_touching": "excluded from overlap",
            "cross_epsg_comparison": "not performed",
            "data": "train metadata only; no test data",
        },
        "inputs": {
            "af_metadata": _file_digest(af_path),
            "bs_metadata": _file_digest(bs_path),
            "split": _file_digest(split_path),
        },
        "tasks": {"af": af, "bs": bs},
        "overall_passes_no_positive_area_overlap": bool(
            af["passes_no_positive_area_overlap"]
            and bs["passes_no_positive_area_overlap"]
        ),
        "privacy": "aggregate counts only; no chip IDs or coordinates",
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit train/validation positive-area overlap of metadata grid bounds; "
            "output contains aggregate counts only"
        )
    )
    parser.add_argument("--af-meta", required=True)
    parser.add_argument("--bs-meta", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    inputs = {Path(value).resolve() for value in (args.af_meta, args.bs_meta, args.split)}
    if output in inputs:
        raise ValueError("output must not overwrite an input")
    payload = build_report(args)
    _write_atomic(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
