"""Replay saved bundle inference on validation features without model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args(argv)
    os.environ["DEVICE"] = args.device
    from competition.data import Chip
    from competition.metrics import competition_score
    from competition.service_bridge import describe_models, predict_features
    from competition.tree import load_records, _read_record

    def deny(*args, **kwargs):
        raise RuntimeError("Network blocked during validation replay")

    socket.create_connection = deny
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    bundle = describe_models(args.model_dir)
    if not bundle["available"]:
        raise RuntimeError(bundle["reason"])
    reference, prediction, digests, timing = {}, {}, {}, {}
    for task in ("af", "bs"):
        rows = [row for row in load_records(args.records, task) if row.split == "val"]
        reference[task], prediction[task] = [], []
        started = time.monotonic()
        digest = hashlib.sha256()
        for index, row in enumerate(rows):
            x, y, valid = _read_record(row)
            chip = Chip(
                row.id, task, x, list(row.feature_names), np.ones(y.shape, bool), {}
            )
            output = predict_features(chip, args.model_dir)
            mask = np.ascontiguousarray(output["class_map"])
            digest.update(row.id.encode("utf-8"))
            digest.update(mask.tobytes(order="C"))
            reference[task].append(np.where(valid, y, 0).astype(np.uint8))
            prediction[task].append(np.where(valid, mask, 0).astype(np.uint8))
            if index % 20 == 0:
                print(f"Replay {task}: {index + 1}/{len(rows)}", flush=True)
        digests[task] = digest.hexdigest()
        timing[task] = {"chips": len(rows), "seconds": time.monotonic() - started}
    metrics = competition_score(
        reference["af"], prediction["af"], reference["bs"], prediction["bs"]
    )
    expected = {
        "F1_af": bundle["tasks"]["af"]["selection"]["F1_af"],
        **{
            name: bundle["tasks"]["bs"]["selection"][name]
            for name in ("IoU_burn", "mIoU_sev")
        },
    }
    delta = {name: abs(metrics[name] - value) for name, value in expected.items()}
    report = {
        "bundle_id": bundle["bundle_id"],
        "device": args.device,
        "network_blocked": True,
        "scope": "Saved-weight runtime replay; not retraining or independent test",
        "metrics": metrics,
        "expected_validation": expected,
        "absolute_delta": delta,
        "tolerance": args.tolerance,
        "within_tolerance": all(value <= args.tolerance for value in delta.values()),
        "class_map_digests": digests,
        "timing": timing,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["within_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
