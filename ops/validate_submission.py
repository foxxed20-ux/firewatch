"""Independently validate a written competition CSV against the official template."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from competition.rle import decode_bs_rles, decode_rle, encode_rle


def read_rows(path):
    csv.field_size_limit(10 * 1024 * 1024)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["chip_id", "class_id", "rle"]:
            raise ValueError("CSV must have the exact header chip_id,class_id,rle")
        rows = list(reader)
    keys = [(row["chip_id"], int(row["class_id"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate chip/class pair")
    return rows, keys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    rows, keys = read_rows(args.submission)
    _, expected = read_rows(args.template)
    if keys != expected:
        raise ValueError("Submission keys/order differ from the official template")
    if len(rows) != 447:
        raise ValueError("This competition requires exactly 447 data rows")
    per_chip = {}
    for row in rows:
        chip, cls, rle = row["chip_id"], int(row["class_id"]), row["rle"]
        if rle is None or rle.lower() in {"nan", "null", "none"}:
            raise ValueError("Missing or NaN RLE")
        shape = (256, 256) if chip.startswith("AF_") else (512, 512)
        decoded = decode_rle(rle, shape)
        if encode_rle(decoded) != rle:
            raise ValueError("Non-canonical RLE")
        per_chip.setdefault(chip, {})[cls] = rle
    for chip, rles in per_chip.items():
        if chip.startswith("AF_"):
            if set(rles) != {1}:
                raise ValueError("AF requires class 1")
        else:
            decode_bs_rles(rles, (512, 512))
    af_count = sum(chip.startswith("AF_") for chip in per_chip)
    bs_count = len(per_chip) - af_count
    if (af_count, bs_count) != (180, 89):
        raise ValueError("Unexpected official test chip counts")
    report = {
        "status": "pass",
        "data_rows": len(rows),
        "header": True,
        "af_chips": af_count,
        "bs_chips": bs_count,
        "template_order_exact": True,
        "canonical_rle": True,
        "bs_classes_exclusive": True,
        "submission_sha256": hashlib.sha256(
            Path(args.submission).read_bytes()
        ).hexdigest(),
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
