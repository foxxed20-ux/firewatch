"""Prepare official train chips using an explicit, auditable split file."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .data import discover_chips, read_chip, read_training_mask, FEATURE_VERSION


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", required=True)
    a = p.parse_args(argv)
    out = Path(a.output)
    (out / "npz").mkdir(parents=True, exist_ok=True)
    split_data = json.loads(Path(a.split).read_text(encoding="utf-8"))
    split = split_data.get("split", split_data.get("assignments", split_data))
    records = []
    start = time.monotonic()
    counts = {}
    inputs = discover_chips(a.data_dir)
    if set(inputs) != set(split):
        raise ValueError(
            f"Split/input mismatch: missing={sorted(set(split) - set(inputs))}, extra={sorted(set(inputs) - set(split))}"
        )
    for i, (chip_id, paths) in enumerate(inputs.items()):
        entry = split[chip_id]
        partition = entry["split"] if isinstance(entry, dict) else entry
        if partition not in ("train", "val"):
            raise ValueError(partition)
        chip = read_chip(chip_id, paths)
        y = read_training_mask(chip_id, paths)
        xfile = f"npz/{chip_id}_x.npz"
        yfile = f"npz/{chip_id}_y.npz"
        np.savez(out / xfile, x=chip.x)
        np.savez(out / yfile, y=y)
        rec = {
            "id": chip_id,
            "task": chip.task,
            "split": partition,
            "x_path": xfile,
            "y_path": yfile,
            "feature_names": chip.feature_names,
        }
        if isinstance(entry, dict):
            rec.update({k: v for k, v in entry.items() if k not in rec})
        records.append(rec)
        counts[chip.task + "_" + partition] = (
            counts.get(chip.task + "_" + partition, 0) + 1
        )
        if i % 20 == 0:
            print(
                "PREPARE",
                i,
                chip_id,
                chip.x.shape,
                round(time.monotonic() - start, 1),
                flush=True,
            )
    (out / "records.json").write_text(json.dumps(records, indent=2))
    report = {
        "feature_version": FEATURE_VERSION,
        "counts": counts,
        "seconds": time.monotonic() - start,
        "split_sha256": hashlib.sha256(Path(a.split).read_bytes()).hexdigest(),
        "cloud_pixels_included": True,
    }
    (out / "preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
