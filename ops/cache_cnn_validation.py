"""Write an atomic FP32 CNN validation-probability cache for offline comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.evaluate import _cnn, _cnn_probability
from competition.tree import _read_record, load_records


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp.npz"
    )
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        np.savez_compressed(temporary, **arrays)
        with np.load(temporary, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise RuntimeError("atomic cache verification failed: key mismatch")
            for key, value in arrays.items():
                if (
                    archive[key].shape != value.shape
                    or archive[key].dtype != value.dtype
                ):
                    raise RuntimeError(f"atomic cache verification failed for {key}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create FP32 validation CNN probabilities without thresholding or tuning"
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--task", choices=("af", "bs"), required=True)
    parser.add_argument("--cnn-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    rows = [
        record
        for record in load_records(args.records, args.task)
        if record.split == "val"
    ]
    if not rows:
        raise ValueError(f"no validation records for {args.task}")
    model_dir = Path(args.cnn_root) / args.task
    checkpoint = model_dir / "best.pt"
    config = model_dir / "config.json"
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError(
            "cnn task directory must contain best.pt and config.json"
        )
    model, state, _ = _cnn(model_dir, args.task)
    device = torch.device(args.device)
    model = model.to(device).eval()
    started = time.monotonic()
    reference, valid, probability, chip_ids = [], [], [], []
    expected_channels = int(state["model"]["in_channels"])
    expected_names = list(state["feature_meta"]["names"])
    for record in rows:
        x, y, observation_valid = _read_record(record)
        if (
            x.shape[0] != expected_channels
            or list(record.feature_names) != expected_names
        ):
            raise ValueError(f"{record.id}: ordered CNN feature contract mismatch")
        raw_probability = _cnn_probability(model, state, device, x, args.task)
        raw_probability = np.asarray(raw_probability, dtype=np.float32)
        expected_shape = x.shape[1:] if args.task == "af" else (*x.shape[1:], 4)
        if (
            raw_probability.shape != expected_shape
            or not np.isfinite(raw_probability).all()
        ):
            raise RuntimeError(f"{record.id}: invalid CNN probability output")
        # _read_record makes y==255 invalid; clouds remain valid and are retained.
        reference.append(np.where(observation_valid, y, 0).astype(np.uint8))
        valid.append(observation_valid.astype(bool, copy=False))
        probability.append(raw_probability)
        chip_ids.append(record.id)
    payload = {
        "reference": np.stack(reference).astype(np.uint8, copy=False),
        "valid": np.stack(valid).astype(bool, copy=False),
        "cnn_probability": np.stack(probability).astype(np.float32, copy=False),
        "chip_ids": np.asarray(chip_ids),
    }
    output = Path(args.output)
    _atomic_npz(output, **payload)
    sidecar = {
        "scope": "validation_only; no thresholding, tuning, or test data",
        "task": args.task,
        "count": len(rows),
        "device": str(device),
        "precision": "float32",
        "checkpoint_sha256": _sha256(checkpoint),
        "config_sha256": _sha256(config),
        "seconds": time.monotonic() - started,
    }
    sidecar_path = output.with_suffix(output.suffix + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **sidecar}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
