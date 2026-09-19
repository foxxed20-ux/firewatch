"""Official entry point: python inference.py --data-dir TEST --output submission.csv."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import time
from competition.data import discover_chips, read_chip
from competition.rle import encode_rle, decode_rle
from competition.service_bridge import predict_features, describe_models


def main(argv=None):
    # A valid fragmented 512x512 severity mask can exceed csv's 128 KiB default.
    csv.field_size_limit(10 * 1024 * 1024)
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-dir")
    p.add_argument("--sample-submission")
    args = p.parse_args(argv)
    started = time.monotonic()
    root = Path(args.data_dir)
    templates = (
        [Path(args.sample_submission)]
        if args.sample_submission
        else list(root.rglob("sample_submission.csv"))
    )
    if len(templates) != 1:
        raise ValueError("Exactly one sample_submission.csv is required")
    with templates[0].open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["chip_id", "class_id", "rle"]:
            raise ValueError("Unexpected submission columns")
        rows = list(reader)
    keys = [(r["chip_id"], int(r["class_id"])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate template rows")
    inputs = discover_chips(root)
    wanted = {c for c, _ in keys}
    if wanted != set(inputs):
        raise ValueError("Template and input chip sets differ")
    expected = {
        (c, k) for c in wanted for k in ([1] if c.startswith("AF_") else [1, 2, 3])
    }
    if set(keys) != expected:
        raise ValueError("Template class rows are incomplete or unexpected")
    results = {}
    stats = []
    for i, chip_id in enumerate(dict.fromkeys(c for c, _ in keys)):
        chip = read_chip(chip_id, inputs[chip_id])
        result = predict_features(chip, args.model_dir)
        mask = result["class_map"]
        for cls in [1] if chip.task == "af" else [1, 2, 3]:
            encoded = encode_rle(mask == cls)
            if not (decode_rle(encoded, mask.shape) == (mask == cls)).all():
                raise RuntimeError("RLE round-trip failed")
            results[(chip_id, cls)] = encoded
        stats.append({"chip_id": chip_id, "positive_pixels": int((mask > 0).sum())})
        if i % 20 == 0:
            print(f"Predicted {i + 1}/{len(wanted)} chips", flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["chip_id", "class_id", "rle"])
        writer.writerows(
            (chip_id, cls, results[(chip_id, cls)]) for chip_id, cls in keys
        )
    tmp.replace(out)
    report = {
        "rows": len(rows),
        "chips": len(wanted),
        "seconds": time.monotonic() - started,
        "bundle_id": describe_models(args.model_dir)["bundle_id"],
        "round_trip_verified": True,
        "stats": stats,
    }
    out.with_suffix(".audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "stats"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
