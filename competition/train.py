"""Reproducible AF/BS trainer over preprocessed NPZ records.

Records JSON is a list (or ``{\"records\": [...]}``) of objects containing
``id, task, split, x_path, y_path`` and optionally ``valid_path`` and
``feature_names``. NPZ uses ``x``/``y``/``valid`` keys (``arr_0`` is accepted).
Targets are AF 0/1 or BS 0..3; 255 is ignore. Validation is never augmented.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .models import build_model
try:
    from .metrics import competition_score, micro_f1_af, micro_iou_burn, micro_iou_severity
except ImportError:  # Allows model shape smoke tests before metrics lands.
    competition_score = micro_f1_af = micro_iou_burn = micro_iou_severity = None

IGNORE = 255


def _npz(path: Path, preferred: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive[preferred] if preferred in archive else archive["arr_0"])


def _path(value: str, root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


@dataclass(frozen=True)
class Record:
    id: str
    task: str
    split: str
    x_path: Path
    y_path: Path
    valid_path: Path | None = None
    feature_names: tuple[str, ...] = ()


def load_records(path: str | Path, task: str | None = None) -> list[Record]:
    manifest = Path(path).resolve()
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    rows = raw["records"] if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("records JSON must be a list or object with 'records'")
    parsed: list[Record] = []
    seen_ids: dict[str, str] = {}
    data_splits: dict[Path, str] = {}
    for i, row in enumerate(rows):
        required = {"id", "task", "split", "x_path", "y_path"}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"record {i} lacks required keys {sorted(required)}")
        row_task = str(row["task"]).lower()
        if row_task not in {"af", "bs"} or str(row["split"]) not in {"train", "val"}:
            raise ValueError(f"record {row['id']!r}: invalid task or split")
        record_id, split = str(row["id"]), str(row["split"])
        if record_id in seen_ids:
            raise ValueError(f"duplicate record id {record_id!r} in {seen_ids[record_id]!r} and {split!r}")
        seen_ids[record_id] = split
        x_path = _path(str(row["x_path"]), manifest.parent).resolve()
        y_path = _path(str(row["y_path"]), manifest.parent).resolve()
        valid_path = _path(str(row["valid_path"]), manifest.parent).resolve() if row.get("valid_path") else None
        # No feature, label, or observation mask may cross train/val under a new id.
        for labeled_path in (x_path, y_path, valid_path):
            if labeled_path is None: continue
            if labeled_path in data_splits and data_splits[labeled_path] != split:
                raise ValueError(f"data path {labeled_path} occurs in both train and val")
            data_splits[labeled_path] = split
        if task and row_task != task:
            continue
        parsed.append(Record(record_id, row_task, split, x_path, y_path, valid_path, tuple(map(str, row.get("feature_names", [])))))
    if not parsed:
        raise ValueError(f"no records for task {task!r}")
    return parsed


def _read_record(record: Record) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _npz(record.x_path, "x").astype(np.float32, copy=False)
    raw_y = _npz(record.y_path, "y")
    if not np.issubdtype(raw_y.dtype, np.integer):
        raise ValueError(f"{record.id}: labels must be integer dtype, got {raw_y.dtype}")
    y = raw_y.astype(np.int64, copy=False)
    raw_valid = _npz(record.valid_path, "valid") if record.valid_path else np.ones_like(y, dtype=bool)
    if raw_valid.dtype != np.bool_:
        if not np.issubdtype(raw_valid.dtype, np.number) or not np.isfinite(raw_valid).all() or not np.isin(raw_valid, (0, 1)).all():
            raise ValueError(f"{record.id}: valid mask must be bool or finite 0/1")
    valid = raw_valid.astype(bool, copy=False)
    if x.ndim != 3 or y.ndim != 2 or x.shape[1:] != y.shape or valid.shape != y.shape:
        raise ValueError(f"{record.id}: expected x CHW and y/valid HW, got {x.shape}/{y.shape}/{valid.shape}")
    high = 1 if record.task == "af" else 3
    if np.any((y != IGNORE) & ((y < 0) | (y > high))):
        raise ValueError(f"{record.id}: labels outside 0..{high},255")
    return x, y.astype(np.uint8, copy=False), valid & (y != IGNORE)


def train_normalizer(records: Iterable[Record]) -> dict[str, list[float]]:
    """Compute per-channel moments on train pixels only, excluding invalid/NaN."""
    total = total_sq = None
    count = None
    for record in records:
        x, _, valid = _read_record(record)
        finite = np.isfinite(x)
        mask = finite & valid[None]
        values = np.where(mask, x, 0.0).astype(np.float64)
        sums = values.sum(axis=(1, 2))
        squares = (values * values).sum(axis=(1, 2))
        amounts = mask.sum(axis=(1, 2), dtype=np.int64)
        if total is None:
            total, total_sq, count = sums, squares, amounts
        else:
            total += sums; total_sq += squares; count += amounts
    if total is None or np.any(count == 0):
        raise ValueError("at least one feature has no finite valid train pixels")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 1e-8))
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist(), "count": count.astype(int).tolist()}


class ChipDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[Record], normalizer: dict[str, list[float]], train: bool, crop_size: int) -> None:
        self.records, self.train, self.crop_size = records, train, crop_size
        self.mean = np.asarray(normalizer["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(normalizer["std"], dtype=np.float32)[:, None, None]

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, valid = _read_record(self.records[index])
        x = np.where(np.isfinite(x), x, self.mean)
        x = (x - self.mean) / self.std
        if self.train and x.shape[1] > self.crop_size:
            top = np.random.randint(0, x.shape[1] - self.crop_size + 1)
            left = np.random.randint(0, x.shape[2] - self.crop_size + 1)
            x, y, valid = x[:, top:top+self.crop_size, left:left+self.crop_size], y[top:top+self.crop_size, left:left+self.crop_size], valid[top:top+self.crop_size, left:left+self.crop_size]
        if self.train:
            if np.random.random() < .5: x, y, valid = x[:, :, ::-1].copy(), y[:, ::-1].copy(), valid[:, ::-1].copy()
            if np.random.random() < .5: x, y, valid = x[:, ::-1, :].copy(), y[::-1, :].copy(), valid[::-1, :].copy()
            turns = np.random.randint(0, 4)
            if turns: x, y, valid = np.rot90(x, turns, axes=(1, 2)).copy(), np.rot90(y, turns).copy(), np.rot90(valid, turns).copy()
        return torch.from_numpy(x.copy()), torch.from_numpy(y.astype(np.int64, copy=False)), torch.from_numpy(valid)


def _binary_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, pos_weight: float) -> torch.Tensor:
    mask = valid.float()
    target_f = torch.where(valid,target,torch.zeros_like(target)).float()
    bce = F.binary_cross_entropy_with_logits(logits[:, 0], target_f, pos_weight=torch.tensor(pos_weight, device=logits.device), reduction="none")
    probability = torch.sigmoid(logits[:, 0])
    focal = ((1 - probability) * target_f + probability * (1 - target_f)).pow(1.5)
    bce = (bce * focal * mask).sum() / mask.sum().clamp_min(1)
    inter = (probability * target_f * mask).sum()
    dice = 1 - (2 * inter + 1) / ((probability * mask).sum() + (target_f * mask).sum() + 1)
    return bce + dice


def _multiclass_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    ce = F.cross_entropy(logits, safe_target, weight=class_weights, reduction="none")
    ce = (ce * valid).sum() / valid.sum().clamp_min(1)
    probability = torch.softmax(logits, dim=1)
    onehot = F.one_hot(safe_target, 4).permute(0, 3, 1, 2).float()
    mask = valid[:, None].float()
    inter = (probability * onehot * mask).sum((0, 2, 3))
    denom = ((probability + onehot) * mask).sum((0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()
    return ce + dice


def _best_af_threshold(reference: list[np.ndarray], probability: list[np.ndarray]) -> tuple[float, float]:
    """Exact global-micro F1 threshold over all validation pixels.

    Sorting unique scores avoids an arbitrary 0.15--0.85 sweep.  The initial
    candidate is above one, representing an intentionally all-background mask.
    """
    truth = np.concatenate([a.reshape(-1) for a in reference]).astype(bool, copy=False)
    score = np.concatenate([a.reshape(-1) for a in probability]).astype(np.float32, copy=False)
    if not np.isfinite(score).all(): raise ValueError("AF validation probabilities must be finite")
    positives = int(truth.sum())
    if not len(score) or positives == 0:
        return float(np.nextafter(1.0, 2.0)), 1.0 if positives == 0 else 0.0
    order = np.argsort(-score, kind="stable"); score, truth = score[order], truth[order]
    # Scores are processed as tied groups. A candidate at a group endpoint
    # predicts exactly scores >= its value. np.argmax retains the first tied
    # maximum, which is the highest threshold as specified.
    endpoints = np.r_[score[:-1] != score[1:], True]
    positions = np.flatnonzero(endpoints)
    true_positive = np.cumsum(truth, dtype=np.int64)[positions]
    predicted = positions + 1
    f1 = 2.0 * true_positive / (predicted + positives)
    winner = int(np.argmax(f1))
    return float(score[positions[winner]]), float(f1[winner])


def _class_balance(records: list[Record], task: str) -> tuple[float, torch.Tensor]:
    counts = np.zeros(2 if task == "af" else 4, dtype=np.int64)
    for r in records:
        _, y, valid = _read_record(r)
        for c in range(len(counts)): counts[c] += int(np.sum(valid & (y == c)))
    if task == "af": return float(np.clip(counts[0] / max(counts[1], 1), 1, 30)), torch.ones(1)
    weights = counts.sum() / np.maximum(counts, 1) / len(counts)
    return 1., torch.tensor(np.clip(weights, .25, 8), dtype=torch.float32)


def _validate(model: nn.Module, loader: DataLoader, task: str, device: torch.device, amp: bool) -> tuple[dict[str, float], float]:
    refs: list[np.ndarray] = []; predictions: list[np.ndarray] = []; probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y, valid in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=amp): logits = model(x)
            raw = torch.sigmoid(logits[:, 0]).float().cpu().numpy() if task == "af" else torch.argmax(logits, 1).cpu().numpy()
            for truth, permitted, predicted in zip(y.numpy(), valid.numpy(), raw):
                # valid is NoData-only; clouds stay valid and are therefore scored.
                if task == "af":
                    # Store permitted pixels only, so a threshold of exactly zero
                    # cannot turn NoData into a false positive.
                    refs.append(truth[permitted].astype(np.uint8))
                    probabilities.append(predicted[permitted])
                else: predictions.append(np.where(permitted, predicted, 0).astype(np.uint8))
                if task != "af": refs.append(np.where(permitted, truth, 0).astype(np.uint8))
    if task == "af":
        best_threshold, best_f1 = _best_af_threshold(refs, probabilities)
        return {"F1_af": best_f1, "threshold": best_threshold}, best_f1
    iou_burn = float(micro_iou_burn(refs, predictions)) if micro_iou_burn else _burn_iou(refs, predictions)
    sev = micro_iou_severity(refs, predictions) if micro_iou_severity else {c: _class_iou(refs, predictions, c) for c in (1, 2, 3)}
    miou = float(sum(sev.values()) / 3)
    return {"IoU_burn": iou_burn, "mIoU_sev": miou, **{f"IoU_{c}": float(sev[c]) for c in (1,2,3)}}, .35 * iou_burn + .30 * miou


def _f1(r: list[np.ndarray], p: list[np.ndarray]) -> float:
    tp = fp = fn = 0
    for a,b in zip(r,p): tp += int(((a==1)&(b==1)).sum()); fp += int(((a==0)&(b==1)).sum()); fn += int(((a==1)&(b==0)).sum())
    return 1. if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn)
def _class_iou(r: list[np.ndarray], p: list[np.ndarray], c: int) -> float:
    inter = union = 0
    for a,b in zip(r,p): inter += int(((a==c)&(b==c)).sum()); union += int(((a==c)|(b==c)).sum())
    return 1. if union == 0 else inter/union
def _burn_iou(r: list[np.ndarray], p: list[np.ndarray]) -> float:
    return _class_iou([a>0 for a in r], [b>0 for b in p], True)


def _save(path: Path, model: nn.Module, task: str, channels: int, base: int, normalizer: dict[str, list[float]], feature_names: tuple[str,...], epoch: int, score: float, metrics: dict[str,float], threshold: float | None) -> None:
    torch.save({"format_version": 1, "model": {"name":"SmallResUNet", "in_channels":channels, "out_channels":1 if task=="af" else 4, "base_channels":base, "task":task}, "model_state": model.state_dict(), "normalizer":normalizer, "feature_meta":{"names":list(feature_names)}, "epoch":epoch, "selection_score":score, "validation":metrics, "threshold":threshold}, path)


def train_task(records: list[Record], output: Path, *, epochs: int, batch_size: int, learning_rate: float, crop_size: int, base_channels: int, time_limit_min: float, seed: int, workers: int) -> dict[str, Any]:
    task = records[0].task
    if any(r.task != task for r in records): raise ValueError("train_task takes one task")
    train_records = [r for r in records if r.split == "train"]; val_records = [r for r in records if r.split == "val"]
    if not train_records or not val_records: raise ValueError(f"{task}: both train and val records are required")
    x0, _, _ = _read_record(train_records[0]); channels = x0.shape[0]
    if any(_read_record(r)[0].shape[0] != channels for r in records): raise ValueError(f"{task}: channel count differs between records")
    feature_names = train_records[0].feature_names
    if not feature_names or len(feature_names) != channels or len(set(feature_names)) != channels:
        raise ValueError(f"{task}: every record requires unique ordered feature_names for all channels")
    if any(r.feature_names != feature_names for r in records):
        raise ValueError(f"{task}: feature_names order differs between records")
    normalizer = train_normalizer(train_records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = device.type == "cuda"
    model = build_model(channels, task, base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and not torch.cuda.is_bf16_supported())
    train_loader = DataLoader(ChipDataset(train_records, normalizer, True, crop_size), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=amp, persistent_workers=workers>0)
    val_loader = DataLoader(ChipDataset(val_records, normalizer, False, crop_size), batch_size=1, shuffle=False, num_workers=workers, pin_memory=amp)
    pos_weight, class_weights = _class_balance(train_records, task); class_weights = class_weights.to(device)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({"task":task,"channels":channels,"feature_names":list(feature_names),"base_channels":base_channels,"seed":seed,"epochs":epochs,"batch_size":batch_size,"learning_rate":learning_rate,"crop_size":crop_size,"time_limit_min":time_limit_min,"workers":workers,"records":len(records),"class_balance":{"pos_weight":pos_weight,"class_weights":class_weights.cpu().tolist()}}, indent=2), encoding="utf-8")
    best = -math.inf; history = []; started = time.monotonic()
    for epoch in range(1, epochs+1):
        if epoch > 1 and time.monotonic() - started >= time_limit_min * 60: break
        model.train(); losses = []
        for x,y,valid in train_loader:
            x,y,valid = x.to(device,non_blocking=True),y.to(device,non_blocking=True),valid.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=amp):
                loss = _binary_loss(model(x),y,valid,pos_weight) if task=="af" else _multiclass_loss(model(x),y,valid,class_weights)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics, selection = _validate(model,val_loader,task,device,amp)
        entry = {"epoch":epoch,"train_loss":float(np.mean(losses)),"selection_score":selection,**metrics}; history.append(entry)
        _save(output / "last.pt",model,task,channels,base_channels,normalizer,feature_names,epoch,selection,metrics,metrics.get("threshold"))
        if selection > best:
            best = selection; _save(output / "best.pt",model,task,channels,base_channels,normalizer,feature_names,epoch,selection,metrics,metrics.get("threshold"))
        (output / "history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        print(json.dumps(entry), flush=True)
    return {"task":task,"best_selection_score":best,"epochs_completed":len(history),"output":str(output)}


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description="Train FireWatch AF/BS segmentation models from preprocessed NPZ chips")
    p.add_argument("--records",required=True); p.add_argument("--output",required=True); p.add_argument("--task",choices=("af","bs","all"),default="all")
    p.add_argument("--epochs",type=int,default=50); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--crop-size",type=int,default=256); p.add_argument("--base-channels",type=int,default=24); p.add_argument("--time-limit-min",type=float,default=55); p.add_argument("--seed",type=int,default=42); p.add_argument("--workers",type=int,default=2)
    a=p.parse_args(argv); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    root=Path(a.output); tasks=("af","bs") if a.task=="all" else (a.task,)
    result=[]
    for task in tasks: result.append(train_task(load_records(a.records,task),root/task,epochs=a.epochs,batch_size=a.batch_size,learning_rate=a.lr,crop_size=a.crop_size,base_channels=a.base_channels,time_limit_min=a.time_limit_min,seed=a.seed,workers=a.workers))
    print(json.dumps(result,indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
