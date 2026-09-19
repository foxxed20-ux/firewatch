"""Prepare only the frozen FireWatch validation chips from the verified TRAIN tar."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.data import (
    FEATURE_VERSION,
    discover_chips,
    read_chip,
    read_training_mask,
)


EXPECTED_TAR_SHA256 = "9cc2d532312dd91d24beb8c6abcfdddac0cff6ca095501be28d81f751a8f159a"
EXPECTED_VAL_COUNTS = {"af": 84, "bs": 45}
INPUT_KEYS = {
    "af": {"viirs", "aux"},
    "bs": {"s2_pre", "s2_post", "s1_pre", "s1_post", "aux"},
}
SUFFIXES = {
    "af": ("VIIRS_I1-I5.tif", "AUX.tif", "MASK.tif"),
    "bs": (
        "Sentinel-2_pre.tif",
        "Sentinel-2_post.tif",
        "Sentinel-1_pre.tif",
        "Sentinel-1_post.tif",
        "AUX.tif",
        "MASK.tif",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp.json"
    )
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, key: str, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp.npz"
    )
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **{key: value})
            stream.flush()
            os.fsync(stream.fileno())
        with np.load(temporary, allow_pickle=False) as archive:
            if archive.files != [key] or archive[key].shape != value.shape or archive[key].dtype != value.dtype:
                raise RuntimeError(f"NPZ verification failed for {path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _split(path: Path) -> tuple[dict[str, str], list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assignments = raw.get("split", raw.get("assignments", raw)) if isinstance(raw, dict) else raw
    if not isinstance(assignments, dict):
        raise ValueError("split must be an object mapping chip_id to train/val")
    parsed: dict[str, str] = {}
    for chip_id, entry in assignments.items():
        partition = entry.get("split") if isinstance(entry, dict) else entry
        if partition not in {"train", "val"}:
            raise ValueError(f"invalid split for {chip_id}: {partition!r}")
        parsed[str(chip_id)] = str(partition)
    selected = sorted(chip_id for chip_id, partition in parsed.items() if partition == "val")
    counts = {
        task: sum(chip_id.startswith(task.upper() + "_") for chip_id in selected)
        for task in ("af", "bs")
    }
    if len(parsed) != 644 or counts != EXPECTED_VAL_COUNTS:
        raise ValueError(f"unexpected frozen split size/counts: all={len(parsed)}, val={counts}")
    return parsed, selected


def _source_inventory(data_dir: Path, selected: list[str]) -> dict[str, Any]:
    inputs = discover_chips(data_dir)
    missing = []
    complete = 0
    for chip_id in selected:
        task = chip_id[:2].lower()
        paths = inputs.get(chip_id, {})
        absent = sorted(INPUT_KEYS[task] - set(paths))
        sample = next(iter(paths.values()), None)
        mask = (
            sample.parent.parent / "masks" / f"{chip_id}_MASK.tif"
            if sample is not None
            else None
        )
        if mask is None or not mask.is_file():
            absent.append("mask")
        if absent:
            missing.append({"chip_id": chip_id, "missing": absent})
        else:
            complete += 1
    return {
        "path": str(data_dir.resolve()),
        "discovered_input_chips": len(inputs),
        "selected_complete_chips": complete,
        "selected_incomplete_chips": len(missing),
        "complete": not missing,
        "missing_examples": missing[:20],
    }


def _expected_basenames(selected: list[str]) -> dict[str, str]:
    expected = {}
    for chip_id in selected:
        task = chip_id[:2].lower()
        for suffix in SUFFIXES[task]:
            name = f"{chip_id}_{suffix}"
            expected[name] = chip_id
    return expected


def _tar_members(archive: tarfile.TarFile, selected: list[str]) -> dict[str, tarfile.TarInfo]:
    expected = _expected_basenames(selected)
    found: dict[str, tarfile.TarInfo] = {}
    for member in archive:
        path = PurePosixPath(member.name)
        if path.name not in expected:
            continue
        if (
            not member.isfile()
            or path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != "train"
        ):
            raise ValueError(f"unsafe selected TRAIN member: {member.name!r}")
        if path.name in found:
            raise ValueError(f"duplicate selected TRAIN file: {path.name}")
        found[path.name] = member
    missing = sorted(set(expected) - set(found))
    if missing:
        raise ValueError(f"TRAIN tar lacks {len(missing)} selected files; first={missing[:5]}")
    return found


def _extract_selected(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    raw_root: Path,
) -> list[dict[str, Any]]:
    manifest = []
    root = raw_root.resolve()
    for index, name in enumerate(sorted(members), 1):
        member = members[name]
        relative = PurePosixPath(member.name)
        target = (raw_root / Path(*relative.parts)).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"selected TRAIN path escapes output: {member.name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"cannot read selected TRAIN member: {member.name}")
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f"{target.stem}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        try:
            with source, temporary.open("wb") as output:
                while True:
                    block = source.read(1 << 20)
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                output.flush()
                os.fsync(output.fileno())
            if size != member.size:
                raise ValueError(f"short extraction for {member.name}: {size} != {member.size}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        extracted_sha = _sha256(target)
        if extracted_sha != digest.hexdigest():
            raise RuntimeError(f"extracted bytes differ from tar stream: {member.name}")
        manifest.append(
            {
                "tar_member": member.name,
                "output": target.relative_to(raw_root.parent).as_posix(),
                "bytes": size,
                "sha256": extracted_sha,
            }
        )
        if index % 100 == 0:
            print(f"EXTRACT {index}/{len(members)}", flush=True)
    return manifest


def _feature_contract(bundle_dir: Path) -> dict[str, list[str]]:
    result = {}
    for task in ("af", "bs"):
        path = bundle_dir / task / "tree" / "metadata.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        names = meta.get("feature_names")
        if meta.get("task") != task or not isinstance(names, list) or len(set(names)) != len(names):
            raise ValueError(f"invalid V3 feature metadata: {path}")
        result[task] = [str(value) for value in names]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the 129 frozen validation chips from a verified official TRAIN tar"
    )
    parser.add_argument("--data-dir", required=True, help="Existing TRAIN root inspected read-only")
    parser.add_argument("--train-tar", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model-bundle", required=True, help="Frozen V3 bundle defining feature order")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-tar-sha256", default=EXPECTED_TAR_SHA256)
    args = parser.parse_args(argv)

    started = time.monotonic()
    data_dir = Path(args.data_dir).resolve()
    tar_path = Path(args.train_tar).resolve()
    split_path = Path(args.split).resolve()
    bundle_dir = Path(args.model_bundle).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"output must be a new directory: {output}")
    if not data_dir.is_dir() or not tar_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("data-dir, train-tar, or split is missing")
    expected_tar_sha = str(args.expected_tar_sha256).lower()
    if len(expected_tar_sha) != 64:
        raise ValueError("expected TRAIN tar SHA-256 must have 64 hex characters")

    _, selected = _split(split_path)
    source_inventory = _source_inventory(data_dir, selected)
    tar_sha = _sha256(tar_path)
    if tar_sha != expected_tar_sha:
        raise ValueError(f"TRAIN tar SHA-256 mismatch: {tar_sha}")
    feature_contract = _feature_contract(bundle_dir)

    print(
        json.dumps(
            {
                "phase": "verified_inputs",
                "tar_sha256": tar_sha,
                "source_complete": source_inventory["complete"],
                "selected": len(selected),
            }
        ),
        flush=True,
    )
    output.mkdir(parents=True)
    raw_root = output / "raw"
    with tarfile.open(tar_path, "r:") as archive:
        members = _tar_members(archive, selected)
    with tarfile.open(tar_path, "r:") as archive:
        # TarInfo offsets collected above are valid for this byte-identical archive.
        raw_manifest = _extract_selected(archive, members, raw_root)
    _atomic_json(output / "raw_manifest.json", raw_manifest)

    extracted_train = raw_root / "train"
    inputs = discover_chips(extracted_train)
    if set(inputs) != set(selected):
        raise ValueError(
            f"extracted chip mismatch: missing={len(set(selected)-set(inputs))}, "
            f"extra={len(set(inputs)-set(selected))}"
        )
    records = []
    npz_manifest = []
    counts = {"af_val": 0, "bs_val": 0}
    for index, chip_id in enumerate(selected, 1):
        chip = read_chip(chip_id, inputs[chip_id])
        target = read_training_mask(chip_id, inputs[chip_id])
        if chip.feature_names != feature_contract[chip.task]:
            raise ValueError(f"{chip_id}: feature order differs from frozen V3")
        x_relative = Path("npz") / f"{chip_id}_x.npz"
        y_relative = Path("npz") / f"{chip_id}_y.npz"
        _atomic_npz(output / x_relative, "x", chip.x.astype(np.float32, copy=False))
        _atomic_npz(output / y_relative, "y", target.astype(np.uint8, copy=False))
        x_sha, y_sha = _sha256(output / x_relative), _sha256(output / y_relative)
        records.append(
            {
                "id": chip_id,
                "task": chip.task,
                "split": "val",
                "x_path": x_relative.as_posix(),
                "y_path": y_relative.as_posix(),
                "feature_names": chip.feature_names,
            }
        )
        npz_manifest.extend(
            [
                {"path": x_relative.as_posix(), "bytes": (output / x_relative).stat().st_size, "sha256": x_sha},
                {"path": y_relative.as_posix(), "bytes": (output / y_relative).stat().st_size, "sha256": y_sha},
            ]
        )
        counts[f"{chip.task}_val"] += 1
        del chip, target
        gc.collect()
        if index % 10 == 0 or index == len(selected):
            print(f"PREPARE {index}/{len(selected)}", flush=True)

    records_path = output / "records.json"
    _atomic_json(records_path, records)
    _atomic_json(output / "npz_manifest.json", npz_manifest)
    raw_total = sum(item["bytes"] for item in raw_manifest)
    npz_total = sum(item["bytes"] for item in npz_manifest)
    report = {
        "status": "pass",
        "scope": "official TRAIN frozen validation only; no test data",
        "feature_version": FEATURE_VERSION,
        "counts": counts,
        "records": {
            "path": records_path.relative_to(output).as_posix(),
            "sha256": _sha256(records_path),
            "count": len(records),
        },
        "split": {"path": str(split_path), "sha256": _sha256(split_path)},
        "train_tar": {
            "path": str(tar_path),
            "bytes": tar_path.stat().st_size,
            "sha256": tar_sha,
            "expected_sha256": expected_tar_sha,
        },
        "existing_source_inventory": source_inventory,
        "extraction": {
            "selected_files": len(raw_manifest),
            "bytes": raw_total,
            "all_extracted_bytes_rehashed_against_tar_stream": True,
            "manifest": "raw_manifest.json",
        },
        "prepared_npz": {
            "files": len(npz_manifest),
            "bytes": npz_total,
            "manifest": "npz_manifest.json",
        },
        "feature_names": feature_contract,
        "model_bundle_manifest_sha256": _sha256(bundle_dir / "manifest.json"),
        "cloud_pixels_included": True,
        "peak_design": "one chip at a time; no full-dataset arrays retained",
        "seconds": time.monotonic() - started,
    }
    _atomic_json(output / "preparation.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
