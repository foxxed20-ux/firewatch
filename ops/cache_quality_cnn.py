"""Cache full-frame FP32 SmallResUNet validation probabilities, optionally with D4 TTA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.models import build_model
from competition.tree import _read_record, load_records


D4 = tuple((turns, mirror) for turns in range(4) for mirror in (False, True))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
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
                if archive[key].shape != value.shape or archive[key].dtype != value.dtype:
                    raise RuntimeError(f"atomic cache verification failed for {key}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _transform(tensor, turns: int, mirror: bool):
    import torch

    transformed = torch.rot90(tensor, turns, dims=(-2, -1))
    return torch.flip(transformed, dims=(-1,)) if mirror else transformed


def _inverse_transform(tensor, turns: int, mirror: bool):
    import torch

    restored = torch.flip(tensor, dims=(-1,)) if mirror else tensor
    return torch.rot90(restored, -turns, dims=(-2, -1))


def _check_d4_inverse() -> None:
    """Fail fast if any inverse is wrong, including on a non-square tensor."""
    import torch

    source = torch.arange(30, dtype=torch.float32).reshape(1, 1, 5, 6)
    variants = set()
    for turns, mirror in D4:
        transformed = _transform(source, turns, mirror)
        restored = _inverse_transform(transformed, turns, mirror)
        if not torch.equal(restored, source):
            raise RuntimeError(f"invalid D4 inverse for turns={turns}, mirror={mirror}")
        variants.add((tuple(transformed.shape), transformed.numpy().tobytes()))
    if len(variants) != 8:
        raise RuntimeError("D4 transform set does not contain eight unique transforms")


def forward_probability(model, x_tensor, task: str, tta: str):
    """Return spatially aligned raw probabilities as an FP32 CPU NCHW tensor.

    D4 members are evaluated sequentially, inverted before accumulation, and
    averaged without class multipliers or decision thresholds.
    """
    import torch

    if task not in {"af", "bs"}:
        raise ValueError("task must be 'af' or 'bs'")
    if tta not in {"none", "d4"}:
        raise ValueError("tta must be 'none' or 'd4'")
    if not isinstance(x_tensor, torch.Tensor) or x_tensor.ndim != 4:
        raise ValueError("x_tensor must be an NCHW torch.Tensor")
    transforms = D4 if tta == "d4" else ((0, False),)
    total = None
    with torch.inference_mode():
        for turns, mirror in transforms:
            logits = model(_transform(x_tensor, turns, mirror))
            probability = (
                torch.sigmoid(logits[:, :1])
                if task == "af"
                else torch.softmax(logits, dim=1)
            )
            restored = _inverse_transform(probability.float(), turns, mirror).cpu()
            total = restored.clone() if total is None else total.add_(restored)
            del logits, probability, restored
        if total is not None:
            total.mul_(1.0 / len(transforms))
    if total is None:
        raise RuntimeError("no TTA transforms were evaluated")
    return total


def _probability(model, normalized: np.ndarray, task: str, device, tta: str) -> np.ndarray:
    import torch

    aligned = forward_probability(
        model, torch.from_numpy(normalized[None]).to(device), task, tta
    )[0]
    return (
        aligned[0].numpy()
        if task == "af"
        else aligned.permute(1, 2, 0).numpy()
    ).astype(np.float32, copy=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cache raw full-frame FP32 CNN validation probabilities"
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=("af", "bs"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tta", choices=("none", "d4"), default="none")
    args = parser.parse_args(argv)

    import torch

    _check_d4_inverse()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    records_path = Path(args.records).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    config_path = checkpoint_path.parent / "config.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config not found: {config_path}")
    if output.suffix.lower() != ".npz":
        raise ValueError("output must use the .npz extension")
    if output in {checkpoint_path, config_path, records_path}:
        raise ValueError("output must not overwrite an input")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint.get("model")
    if not isinstance(model_config, dict) or model_config.get("task") != args.task:
        raise ValueError("checkpoint task/model metadata mismatch")
    channels = int(model_config["in_channels"])
    base_channels = int(model_config["base_channels"])
    expected_names = list(checkpoint["feature_meta"]["names"])
    if len(expected_names) != channels or len(set(expected_names)) != channels:
        raise ValueError("checkpoint feature metadata is invalid")
    mean = np.asarray(checkpoint["normalizer"]["mean"], np.float32)
    std = np.asarray(checkpoint["normalizer"]["std"], np.float32)
    if (
        mean.shape != (channels,)
        or std.shape != (channels,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("checkpoint normalizer is invalid")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("task") != args.task
        or int(config.get("channels", -1)) != channels
        or int(config.get("base_channels", -1)) != base_channels
        or list(config.get("feature_names", [])) != expected_names
    ):
        raise ValueError("checkpoint and sibling config.json metadata differ")
    mean_chw, std_chw = mean[:, None, None], std[:, None, None]
    model = build_model(channels, args.task, base_channels)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()

    rows = [row for row in load_records(records_path, args.task) if row.split == "val"]
    if not rows:
        raise ValueError(f"no validation records for {args.task}")
    reference: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    chip_ids: list[str] = []
    started = time.monotonic()
    for row in rows:
        x, y, permitted = _read_record(row)
        if x.shape[0] != channels or list(row.feature_names) != expected_names:
            raise ValueError(f"{row.id}: ordered CNN feature contract mismatch")
        normalized = (
            np.where(np.isfinite(x), x, mean_chw) - mean_chw
        ) / std_chw
        probability = _probability(
            model, normalized.astype(np.float32, copy=False), args.task, device, args.tta
        )
        expected_shape = x.shape[1:] if args.task == "af" else (*x.shape[1:], 4)
        if probability.shape != expected_shape or not np.isfinite(probability).all():
            raise RuntimeError(f"{row.id}: invalid CNN probability output")
        reference.append(np.where(permitted, y, 0).astype(np.uint8))
        valid.append(permitted.astype(bool, copy=False))
        probabilities.append(probability.astype(np.float32, copy=False))
        chip_ids.append(row.id)
    elapsed = time.monotonic() - started
    payload = {
        "reference": np.stack(reference).astype(np.uint8, copy=False),
        "valid": np.stack(valid).astype(bool, copy=False),
        "cnn_probability": np.stack(probabilities).astype(np.float32, copy=False),
        "chip_ids": np.asarray(chip_ids),
    }
    _atomic_npz(output, **payload)
    sidecar = {
        "scope": "validation_only; raw probabilities; no multipliers, thresholds, tuning, or test data",
        "task": args.task,
        "count": len(rows),
        "device": str(device),
        "precision": "float32",
        "tta": args.tta,
        "tta_transforms": 8 if args.tta == "d4" else 1,
        "seconds": elapsed,
        "records": {"path": str(records_path), "sha256": _sha256(records_path)},
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "selection_score": checkpoint.get("selection_score"),
        },
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "model": {
            "in_channels": channels,
            "base_channels": base_channels,
            "feature_names": expected_names,
        },
        "output_sha256": _sha256(output),
    }
    sidecar_path = output.with_suffix(output.suffix + ".json")
    _atomic_json(sidecar_path, sidecar)
    print(json.dumps({"output": str(output), "sidecar": str(sidecar_path), **sidecar}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
