"""Losslessly recompress stored feature NPZ files with atomic replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def _resolve(value, root):
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def selected_paths(records_path, task, split):
    manifest = Path(records_path).resolve()
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    rows = raw.get("records", raw) if isinstance(raw, dict) else raw
    paths = []
    for row in rows:
        if row.get("task") == task and row.get("split") == split:
            paths.append(_resolve(row["x_path"], manifest.parent).resolve())
    return list(dict.fromkeys(paths))


def _fingerprint(array):
    if array.dtype.hasobject:
        raise ValueError("object arrays are not permitted in feature cache")
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": tuple(array.shape),
        "sha256_c_order": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _read_fingerprints(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: _fingerprint(archive[key]) for key in archive.files}


def _is_stored_npz(path):
    with zipfile.ZipFile(path) as archive:
        return bool(archive.infolist()) and all(
            item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()
        )


def _write_deflated(source, destination, compression_level):
    with (
        np.load(source, allow_pickle=False) as arrays,
        zipfile.ZipFile(
            destination,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=compression_level,
        ) as archive,
    ):
        for key in arrays.files:
            value = arrays[key]
            if value.dtype.hasobject:
                raise ValueError(f"{source}: object array {key!r} is unsafe")
            with archive.open(f"{key}.npy", mode="w") as member:
                np.lib.format.write_array(member, value, allow_pickle=False)


def compress_one(path, compression_level, apply):
    path = Path(path)
    old_bytes = path.stat().st_size
    if not _is_stored_npz(path):
        return {
            "path": str(path),
            "status": "skipped_already_compressed",
            "old_bytes": old_bytes,
        }
    before = _read_fingerprints(path)
    if not apply:
        return {
            "path": str(path),
            "status": "dry_run_stored",
            "old_bytes": old_bytes,
            "keys": sorted(before),
        }
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp.npz"
    )
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        _write_deflated(path, temporary, compression_level)
        after = _read_fingerprints(temporary)
        if before != after:
            raise RuntimeError(f"bitwise verification failed for {path}")
        new_bytes = temporary.stat().st_size
        temporary.replace(path)
        return {
            "path": str(path),
            "status": "replaced_bitwise_verified",
            "old_bytes": old_bytes,
            "new_bytes": new_bytes,
            "keys": sorted(before),
        }
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Losslessly recompress stored NPZ feature caches; dry-run is the default"
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--task", choices=("af", "bs"), required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.compression_level <= 9:
        raise ValueError("compression-level must be in 0..9")
    report = {
        "apply": args.apply,
        "task": args.task,
        "split": args.split,
        "compression_level": args.compression_level,
        "files": [],
    }
    for path in selected_paths(args.records, args.task, args.split):
        if not path.is_file() or path.suffix.lower() != ".npz":
            report["files"].append(
                {"path": str(path), "status": "skipped_missing_or_not_npz"}
            )
            continue
        report["files"].append(compress_one(path, args.compression_level, args.apply))
    files = report["files"]
    report["processed"] = sum(
        item["status"] == "replaced_bitwise_verified" for item in files
    )
    report["skipped"] = len(files) - report["processed"]
    report["old_bytes"] = sum(item.get("old_bytes", 0) for item in files)
    report["new_bytes"] = sum(item.get("new_bytes", 0) for item in files)
    report["bitwise_verified"] = report["processed"]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
