"""GPU XGBoost diagnostic candidate; it is not supported by the runtime bundle."""

from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
from .metrics import micro_f1_af, micro_iou_burn, micro_iou_severity
from .tree import (
    _best_af_threshold,
    _feature_names,
    _read_record,
    _sample_train,
    load_records,
)


def _xgb():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is required for this optional candidate") from exc
    return xgb


def _probability(booster, x, task, feature_names, chunk=65_536):
    """CPU prediction after GPU training; output is exactly float32 HWC/HW."""
    xgb = _xgb()
    booster.set_param({"device": "cpu"})
    flat = x.reshape(x.shape[0], -1).T
    pieces = []
    for start in range(0, len(flat), chunk):
        part = np.array(flat[start : start + chunk], np.float32, copy=True)
        part[~np.isfinite(part)] = np.nan
        pieces.append(
            booster.predict(xgb.DMatrix(part, feature_names=list(feature_names)))
        )
    p = np.concatenate(pieces)
    if task == "af":
        return p.astype(np.float32, copy=False).reshape(x.shape[1:])
    return p.astype(np.float32, copy=False).reshape(*x.shape[1:], 4)


def _full_validation(booster, rows, task, feature_names):
    refs = []
    valid = []
    probs = []
    for row in rows:
        x, y, v = _read_record(row)
        refs.append(np.where(v, y, 0).astype(np.uint8))
        valid.append(v)
        probs.append(_probability(booster, x, task, feature_names))
    if task == "af":
        threshold, score = _best_af_threshold(
            [r[v] for r, v in zip(refs, valid)], [p[v] for p, v in zip(probs, valid)]
        )
        pred = [
            np.where(v, p >= threshold, 0).astype(np.uint8)
            for p, v in zip(probs, valid)
        ]
        return {
            "F1_af": float(micro_f1_af(refs, pred)),
            "threshold": float(threshold),
        }, float(score)
    best = -1.0
    bm = {}
    bw = None
    for values in itertools.product((0.7, 1.0, 1.3), repeat=3):
        weights = np.asarray((1.0,) + values, np.float32)
        pred = [
            np.where(v, (p * weights).argmax(-1), 0).astype(np.uint8)
            for p, v in zip(probs, valid)
        ]
        burn = float(micro_iou_burn(refs, pred))
        sev = micro_iou_severity(refs, pred)
        mean = float(sum(sev.values()) / 3)
        score = 0.35 * burn + 0.30 * mean
        if score > best:
            best, bw = score, weights
            bm = {
                "IoU_burn": burn,
                "mIoU_sev": mean,
                **{f"IoU_{c}": float(sev[c]) for c in (1, 2, 3)},
            }
    return {**bm, "class_multipliers": bw.astype(float).tolist()}, float(best)


def predict_xgb_features(x, model_dir, feature_names, chunk_pixels=65_536):
    root = Path(model_dir)
    meta = json.loads((root / "metadata.json").read_text())
    if (
        list(feature_names) != list(meta["feature_names"])
        or x.shape[0] != meta["channels"]
    ):
        raise ValueError("ordered feature names or channel count mismatch")
    booster = _xgb().Booster(model_file=str(root / "model.json"))
    p = _probability(booster, x, meta["task"], meta["feature_names"], chunk_pixels)
    if meta["task"] == "af":
        return (p >= meta["threshold"]).astype(np.uint8), p
    return (p * np.asarray(meta["class_multipliers"], np.float32)).argmax(-1).astype(
        np.uint8
    ), p


def train_xgb_task(rows, output, max_pixels, rounds, device, num_threads, seed):
    task = rows[0].task
    train = [r for r in rows if r.split == "train"]
    val = [r for r in rows if r.split == "val"]
    if not train or not val:
        raise ValueError("both train and val records required")
    x, y, w, info = _sample_train(train, task, max_pixels, seed)
    w = (w / w.mean()).astype(np.float32)
    channels = x.shape[1]
    names = _feature_names(rows, channels)
    xgb = _xgb()
    params = {
        "tree_method": "hist",
        "device": device,
        "max_bin": 128,
        "max_depth": 6 if task == "af" else 7,
        "eta": 0.07,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 3.0,
        "min_child_weight": 5.0,
        "seed": seed,
        "nthread": num_threads,
        "objective": "binary:logistic" if task == "af" else "multi:softprob",
        "eval_metric": "logloss" if task == "af" else "mlogloss",
    }
    if task == "bs":
        params["num_class"] = 4
    dtrain = xgb.QuantileDMatrix(x, label=y, weight=w, feature_names=names, max_bin=128)
    # Bounded, stratified validation is only for early stopping; selection is full chips.
    vx, vy, vw, _ = _sample_train(
        val, task, min(200_000, max_pixels // 4), seed + 1, required_classes=False
    )
    vw = (vw / vw.mean()).astype(np.float32)
    dval = xgb.QuantileDMatrix(
        vx, label=vy, weight=vw, feature_names=names, max_bin=128, ref=dtrain
    )
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=25,
    )
    best_iteration = int(getattr(booster, "best_iteration", rounds - 1))
    best_booster = booster[: best_iteration + 1]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    best_booster.save_model(output / "model.json")
    metrics, score = _full_validation(best_booster, val, task, names)
    meta = {
        "format_version": 1,
        "backend": "xgboost",
        "diagnostic_only": True,
        "runtime_supported": False,
        "xgboost_version": xgb.__version__,
        "xgboost_requirement": "xgboost==3.4.1",
        "task": task,
        "channels": channels,
        "feature_names": names,
        "best_iteration": best_iteration,
        "selection_score": score,
        "full_validation": metrics,
        "threshold": metrics.get("threshold"),
        "class_multipliers": metrics.get("class_multipliers", [1.0]),
        "sample": {**info, "weight_mean_normalized": True},
        "train_records": len(train),
        "val_records": len(val),
        "seed": seed,
        "params": params,
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Optional GPU XGBoost FireWatch candidate; validation-only selection"
    )
    p.add_argument("--records", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task", choices=("af", "bs", "all"), default="all")
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--af-max-pixels", type=int, default=800_000)
    p.add_argument("--bs-max-pixels", type=int, default=1_000_000)
    a = p.parse_args(argv)
    result = []
    for task in ("af", "bs") if a.task == "all" else (a.task,):
        result.append(
            train_xgb_task(
                load_records(a.records, task),
                Path(a.output) / task,
                a.af_max_pixels if task == "af" else a.bs_max_pixels,
                a.rounds,
                a.device,
                a.num_threads,
                a.seed,
            )
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
