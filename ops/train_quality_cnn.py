"""Short, validation-only selected BS fine-tune using the deployed CNN architecture.

This is intentionally a separate candidate trainer: it never modifies the
frozen baseline checkpoint, data split, or production training module.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition.models import build_model
from competition.train import (
    _class_balance,
    _multiclass_loss,
    _read_record,
    _validate,
    load_records,
)


class PositiveCropDataset(Dataset):
    """Normalised BS crops; 70% are centred on a uniformly chosen burn class."""

    def __init__(self, records, normalizer, crop_size, seed):
        self.records, self.crop_size = records, crop_size
        self.mean = np.asarray(normalizer["mean"], np.float32)[:, None, None]
        self.std = np.asarray(normalizer["std"], np.float32)[:, None, None]
        self.seed = seed
        self.rng = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.rng is None:
            info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng(info.seed if info else self.seed)
        x, y, valid = _read_record(self.records[index])
        x = (np.where(np.isfinite(x), x, self.mean) - self.mean) / self.std
        height, width = y.shape
        if height > self.crop_size:
            positives = [np.flatnonzero((y == cls) & valid) for cls in (1, 2, 3)]
            present = [pixels for pixels in positives if len(pixels)]
            if present and self.rng.random() < 0.7:
                pixel = int(
                    self.rng.choice(present[int(self.rng.integers(len(present)))])
                )
                row, col = divmod(pixel, width)
                top = int(
                    np.clip(row - self.crop_size // 2, 0, height - self.crop_size)
                )
                left = int(
                    np.clip(col - self.crop_size // 2, 0, width - self.crop_size)
                )
            else:
                top = int(self.rng.integers(height - self.crop_size + 1))
                left = int(self.rng.integers(width - self.crop_size + 1))
            x, y, valid = (
                x[:, top : top + self.crop_size, left : left + self.crop_size],
                y[top : top + self.crop_size, left : left + self.crop_size],
                valid[top : top + self.crop_size, left : left + self.crop_size],
            )
        if self.rng.random() < 0.5:
            x, y, valid = x[:, :, ::-1].copy(), y[:, ::-1].copy(), valid[:, ::-1].copy()
        if self.rng.random() < 0.5:
            x, y, valid = x[:, ::-1, :].copy(), y[::-1, :].copy(), valid[::-1, :].copy()
        turns = int(self.rng.integers(4))
        if turns:
            x, y, valid = (
                np.rot90(x, turns, axes=(1, 2)).copy(),
                np.rot90(y, turns).copy(),
                np.rot90(valid, turns).copy(),
            )
        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.astype(np.int64)),
            torch.from_numpy(valid),
        )


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            self.state[key].lerp_(value.detach(), 1.0 - self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.state, strict=True)


def _save(path, model, source, epoch, score, metrics, config):
    checkpoint = copy.deepcopy(source)
    checkpoint.pop("selection_score", None)
    checkpoint.pop("validation", None)
    checkpoint["model_state"] = model.state_dict()
    checkpoint.update(
        {
            "epoch": epoch,
            "selection_score": score,
            "validation": metrics,
            "quality_config": config,
        }
    )
    torch.save(checkpoint, path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Short BS CNN fine-tune candidate; validation selection only"
    )
    parser.add_argument("--records", required=True)
    parser.add_argument(
        "--source", required=True, help="existing BS directory containing best.pt"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--time-limit-min", type=float, default=42)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    started = time.monotonic()
    if (
        args.crop_size not in {256, 384, 512}
        or args.epochs < 1
        or args.time_limit_min <= 0
    ):
        raise ValueError("invalid crop/epoch/time configuration")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    device = torch.device(device_name)
    source = Path(args.source)
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["model"]["task"] != "bs":
        raise ValueError("source checkpoint is not BS")
    records = load_records(args.records, "bs")
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    names = list(checkpoint["feature_meta"]["names"])
    if any(list(r.feature_names) != names for r in records):
        raise ValueError("feature order differs from source checkpoint")
    model = build_model(
        checkpoint["model"]["in_channels"], "bs", checkpoint["model"]["base_channels"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    ema = EMA(model, args.ema_decay)
    normalizer = checkpoint["normalizer"]
    train_loader = DataLoader(
        PositiveCropDataset(train, normalizer, args.crop_size, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    from competition.train import ChipDataset

    val_loader = DataLoader(
        ChipDataset(val, normalizer, False, args.crop_size),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    stored_weights = source_config.get("class_balance", {}).get("class_weights")
    if stored_weights is None:
        _, class_weights = _class_balance(train, "bs")
    else:
        class_weights = torch.tensor(stored_weights, dtype=torch.float32)
        if class_weights.shape != (4,) or not torch.isfinite(class_weights).all():
            raise ValueError("invalid source class_balance weights")
    class_weights = class_weights.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp and not torch.cuda.is_bf16_supported()
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best_student = best_ema = -math.inf
    history = []
    quality_config = {
        key: getattr(args, key)
        for key in (
            "epochs",
            "time_limit_min",
            "crop_size",
            "batch_size",
            "lr",
            "ema_decay",
            "workers",
            "seed",
            "device",
        )
    }
    (out / "config.json").write_text(
        json.dumps(
            {
                "task": "bs",
                "channels": checkpoint["model"]["in_channels"],
                "feature_names": names,
                "base_channels": checkpoint["model"]["base_channels"],
                "source_checkpoint": str((source / "best.pt").resolve()),
                "source_epoch": checkpoint.get("epoch"),
                "quality_train_command": quality_config,
                "class_balance": {"class_weights": class_weights.cpu().tolist()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for epoch in range(1, args.epochs + 1):
        if time.monotonic() - started >= args.time_limit_min * 60:
            break
        model.train()
        losses = []
        for x, y, valid in train_loader:
            x, y, valid = (
                x.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
                valid.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16,
                enabled=amp,
            ):
                loss = _multiclass_loss(model(x), y, valid, class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        raw_state = copy.deepcopy(model.state_dict())
        student_metrics, student_score = _validate(
            model, val_loader, "bs", device, False
        )
        if student_score > best_student:
            best_student = student_score
            _save(
                out / "best_student.pt",
                model,
                checkpoint,
                epoch,
                student_score,
                student_metrics,
                quality_config,
            )
        ema.copy_to(model)
        metrics, score = _validate(model, val_loader, "bs", device, False)
        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "ema_selection_score": score,
            "student_selection_score": student_score,
            "student_validation": student_metrics,
            **metrics,
        }
        history.append(entry)
        if score > best_ema:
            best_ema = score
            _save(
                out / "best_ema.pt",
                model,
                checkpoint,
                epoch,
                score,
                metrics,
                quality_config,
            )
        model.load_state_dict(raw_state, strict=True)
        _save(out / "last.pt", model, checkpoint, epoch, score, metrics, quality_config)
        (out / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(entry), flush=True)
    print(
        json.dumps(
            {
                "epochs_completed": len(history),
                "best_student_selection_score": best_student,
                "best_ema_selection_score": best_ema,
                "output": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
