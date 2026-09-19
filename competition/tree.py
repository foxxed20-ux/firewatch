"""Fast CPU LightGBM pixel baselines for the FireWatch competition.

The input contract is identical to :mod:`competition.train`: NPZ CHW features,
with labels in a separate NPZ and records split into train/val.  Test chips are
never read here.  Models are saved as LightGBM text plus inspectable JSON.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import micro_f1_af, micro_iou_burn, micro_iou_severity

IGNORE = 255

@dataclass(frozen=True)
class Record:
    id: str; task: str; split: str; x_path: Path; y_path: Path
    valid_path: Path | None = None; feature_names: tuple[str, ...] = ()

def _npz(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z: return np.asarray(z[key] if key in z else z["arr_0"])

def load_records(path: str | Path, task: str | None = None) -> list[Record]:
    manifest=Path(path).resolve(); raw=json.loads(manifest.read_text(encoding="utf-8")); rows=raw["records"] if isinstance(raw,dict) else raw
    result=[]; ids=set(); paths: dict[Path,str]={}
    for row in rows:
        row_task=str(row["task"]).lower()
        record_id,split=str(row["id"]),str(row["split"])
        if row_task not in {"af","bs"} or split not in {"train","val"}: raise ValueError("invalid task or split")
        if record_id in ids: raise ValueError(f"duplicate record id {record_id!r}")
        ids.add(record_id)
        resolve=lambda value: (Path(value) if Path(value).is_absolute() else (manifest.parent / value)).resolve()
        x_path,y_path=resolve(str(row["x_path"])),resolve(str(row["y_path"])); valid_path=resolve(str(row["valid_path"])) if row.get("valid_path") else None
        for data_path in (x_path,y_path,valid_path):
            if data_path is None: continue
            if data_path in paths and paths[data_path] != split: raise ValueError(f"data path {data_path} occurs in both train and val")
            paths[data_path]=split
        if task and row_task != task: continue
        result.append(Record(record_id,row_task,split,x_path,y_path,valid_path,tuple(map(str,row.get("feature_names",[])))))
    if not result: raise ValueError(f"no records for task {task!r}")
    return result

def _read_record(record: Record) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    x=_npz(record.x_path,"x").astype(np.float32,copy=False); raw_y=_npz(record.y_path,"y")
    if x.ndim != 3 or raw_y.ndim != 2 or not np.issubdtype(raw_y.dtype,np.integer): raise ValueError(f"invalid NPZ shapes/types for {record.id}")
    y=raw_y.astype(np.int64,copy=False); raw_valid=_npz(record.valid_path,"valid") if record.valid_path else np.ones_like(y,dtype=bool)
    if raw_valid.dtype != np.bool_ and (not np.issubdtype(raw_valid.dtype,np.number) or not np.isfinite(raw_valid).all() or not np.isin(raw_valid,(0,1)).all()): raise ValueError(f"invalid valid mask for {record.id}")
    valid=raw_valid.astype(bool,copy=False)
    high=1 if record.task=="af" else 3
    if x.shape[1:] != y.shape or valid.shape != y.shape or np.any((y != IGNORE)&((y<0)|(y>high))): raise ValueError(f"invalid labels/shapes for {record.id}")
    return x,y.astype(np.uint8,copy=False),valid&(y!=IGNORE)

def _best_af_threshold(reference: list[np.ndarray], probability: list[np.ndarray]) -> tuple[float,float]:
    truth=np.concatenate([a.reshape(-1) for a in reference]).astype(bool,copy=False); score=np.concatenate([a.reshape(-1) for a in probability]).astype(np.float32,copy=False)
    if not np.isfinite(score).all(): raise ValueError('AF validation probabilities must be finite')
    positives=int(truth.sum())
    if not len(score) or positives==0: return float(np.nextafter(1.,2.)),1. if positives==0 else 0.
    order=np.argsort(-score,kind='stable'); score,truth=score[order],truth[order]
    endpoints=np.r_[score[:-1]!=score[1:],True]; positions=np.flatnonzero(endpoints); tp=np.cumsum(truth,dtype=np.int64)[positions]; predicted=positions+1
    f1=2.*tp/(predicted+positives); winner=int(np.argmax(f1))
    return float(score[positions[winner]]),float(f1[winner])


def _lgb():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required; install the pinned competition environment") from exc
    return lgb


def _feature_names(records: list[Record], channels: int) -> list[str]:
    names = records[0].feature_names
    if not names or len(names) != channels or len(set(names)) != channels:
        raise ValueError("every record requires unique ordered feature_names for all channels")
    if any(tuple(r.feature_names) != names for r in records):
        raise ValueError("feature names differ within one task")
    return list(names)


def _flat(x: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CHW -> valid pixel rows and flat pixel positions without a full transpose copy."""
    positions = np.flatnonzero(valid.reshape(-1))
    return x.reshape(x.shape[0], -1)[:, positions].T.astype(np.float32, copy=False), positions


def _sample_train(records: list[Record], task: str, max_pixels: int, seed: int, required_classes: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Take a bounded, per-chip stratified sample and correct its class prior.

    Each selected class pixel receives ``population_count / selected_count``.
    This retains the original train prior in the boosting objective, while each
    chip still contributes rare classes to the learnable sample.
    """
    rng = np.random.default_rng(seed)
    classes = 2 if task == "af" else 4
    per_chip_class = max(1, max_pixels // max(1, len(records) * classes))
    xs: list[np.ndarray] = []; ys: list[np.ndarray] = []
    population = np.zeros(classes, dtype=np.int64); selected = np.zeros(classes, dtype=np.int64)
    channels: int | None = None
    for r in records:
        x, y, valid = _read_record(r)
        channels = x.shape[0] if channels is None else channels
        if x.shape[0] != channels: raise ValueError("channel count differs within task")
        for c in range(classes):
            locations = np.flatnonzero((valid & (y == c)).reshape(-1))
            population[c] += len(locations)
            if len(locations) > per_chip_class:
                locations = rng.choice(locations, per_chip_class, replace=False)
            if len(locations):
                part = x.reshape(x.shape[0], -1)[:, locations].T.astype(np.float32, copy=False)
                xs.append(part); ys.append(np.full(len(locations), c, dtype=np.uint8)); selected[c] += len(locations)
    if not xs or (required_classes and np.any(selected == 0)):
        missing = np.flatnonzero(selected == 0).tolist()
        raise ValueError(f"training split lacks labels for classes {missing}")
    features = np.concatenate(xs); labels = np.concatenate(ys)
    # LightGBM accepts NaN, but +/-inf must not enter a serialized model.
    features[~np.isfinite(features)] = np.nan
    prior_weights = np.divide(population, selected, out=np.zeros_like(population,dtype=float), where=selected>0)
    weights = prior_weights[labels].astype(np.float32)
    info = {"sampled_pixels":int(len(labels)), "population_counts":population.astype(int).tolist(), "sampled_counts":selected.astype(int).tolist(), "per_chip_per_class_cap":per_chip_class, "prior_weights":prior_weights.astype(float).tolist()}
    return features, labels, weights, info


def _sample_validation(records: list[Record], task: str, max_pixels: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Early stopping needs a bounded eval set. Model selection remains full-chip below.
    x, y, w, _ = _sample_train(records, task, max_pixels, seed, required_classes=False)
    return x, y, w


def _predict_proba(booster: Any, x: np.ndarray, chunk_pixels: int) -> np.ndarray:
    if x.ndim != 3: raise ValueError(f"x must be CHW, got {x.shape}")
    flat = x.reshape(x.shape[0], -1).T
    pieces = []
    for offset in range(0, len(flat), chunk_pixels):
        part = np.array(flat[offset:offset + chunk_pixels], dtype=np.float32, copy=True)
        part[~np.isfinite(part)] = np.nan
        pieces.append(booster.predict(part))
    output = np.concatenate(pieces, axis=0)
    if output.ndim == 1: output = output[:, None]
    return output.reshape(x.shape[1], x.shape[2], output.shape[1])


def predict_tree_features(x: np.ndarray, model_dir: str | Path, chunk_pixels: int = 65_536, feature_names: list[str] | tuple[str,...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(class_map, probabilities)`` from CHW features in bounded batches."""
    lgb = _lgb(); root = Path(model_dir)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if x.shape[0] != int(meta["channels"]):
        raise ValueError(f"model expects {meta['channels']} channels, got {x.shape[0]}")
    if feature_names is None or list(feature_names) != list(meta["feature_names"]):
        raise ValueError("feature_names must exactly match the model's ordered feature metadata")
    probability = _predict_proba(lgb.Booster(model_file=str(root / "model.txt")), x, chunk_pixels)
    if meta["task"] == "af":
        score = probability[..., 0]
        return (score >= float(meta["threshold"])).astype(np.uint8), score
    weights = np.asarray(meta["class_multipliers"], dtype=np.float32)
    if weights.shape != (4,): raise ValueError("invalid class_multipliers in metadata")
    adjusted = probability * weights
    return adjusted.argmax(axis=-1).astype(np.uint8), probability


def _score_bs(reference: list[np.ndarray], prediction: list[np.ndarray]) -> tuple[dict[str, float], float]:
    burn = float(micro_iou_burn(reference, prediction)); severity = micro_iou_severity(reference, prediction)
    mean = float(sum(severity.values()) / 3)
    return {"IoU_burn":burn, "mIoU_sev":mean, **{f"IoU_{c}":float(severity[c]) for c in (1,2,3)}}, .35 * burn + .30 * mean


def _validation_probabilities(booster: Any, records: list[Record]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Run expensive full-chip validation inference once, then reuse it for tuning."""
    reference: list[np.ndarray] = []; probability: list[np.ndarray] = []; permitted: list[np.ndarray] = []
    for r in records:
        x, y, valid = _read_record(r)
        predicted_probability = _predict_proba(booster, x, 65_536)
        # The only ignored pixels are actual NoData. Clouds must have valid=True.
        reference.append(np.where(valid, y, 0).astype(np.uint8)); probability.append(predicted_probability); permitted.append(valid)
    return reference, probability, permitted


def _full_validation(reference: list[np.ndarray], probability: list[np.ndarray], permitted: list[np.ndarray], task: str, *, threshold: float = .5, multipliers: np.ndarray | None = None) -> tuple[dict[str, float], float]:
    if task == "af":
        prediction = [np.where(valid, item[..., 0] >= threshold, 0).astype(np.uint8) for item,valid in zip(probability,permitted)]
        f1 = float(micro_f1_af(reference, prediction)); return {"F1_af":f1, "threshold":threshold}, f1
    if multipliers is None: raise ValueError("BS requires class multipliers")
    prediction = [np.where(valid, (item * multipliers).argmax(axis=-1), 0).astype(np.uint8) for item,valid in zip(probability,permitted)]
    return _score_bs(reference, prediction)


def train_tree_task(records: list[Record], output: str | Path, *, max_pixels: int = 800_000, val_pixels: int = 200_000, num_threads: int = 2, seed: int = 42, rounds: int = 1000) -> dict[str, Any]:
    task = records[0].task
    if any(r.task != task for r in records): raise ValueError("train_tree_task takes a single task")
    train_rows = [r for r in records if r.split == "train"]; val_rows = [r for r in records if r.split == "val"]
    if not train_rows or not val_rows: raise ValueError("both train and val splits are required")
    x, y, weight, sample_info = _sample_train(train_rows, task, max_pixels, seed)
    val_x, val_y, val_weight = _sample_validation(val_rows, task, val_pixels, seed + 1)
    channels = x.shape[1]; names = _feature_names(records, channels); lgb = _lgb()
    params: dict[str, Any] = {"objective":"binary" if task=="af" else "multiclass", "metric":"binary_logloss" if task=="af" else "multi_logloss", "learning_rate":.06, "num_leaves":63, "max_depth":-1, "min_data_in_leaf":80, "feature_fraction":.8, "bagging_fraction":.8, "bagging_freq":1, "lambda_l2":1.0, "verbosity":-1, "seed":seed, "feature_fraction_seed":seed, "bagging_seed":seed, "num_threads":num_threads}
    if task == "bs": params["num_class"] = 4
    train_set=lgb.Dataset(x,label=y,weight=weight,feature_name=names,free_raw_data=True)
    valid_set=lgb.Dataset(val_x,label=val_y,weight=val_weight,reference=train_set,feature_name=names,free_raw_data=True)
    booster=lgb.train(params,train_set,num_boost_round=rounds,valid_sets=[valid_set],callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(50)])
    output=Path(output); output.mkdir(parents=True,exist_ok=True); booster.save_model(str(output / "model.txt"), num_iteration=booster.best_iteration)
    validation_reference, validation_probability, validation_valid = _validation_probabilities(booster, val_rows)
    if task == "af":
        af_ref=[item[valid] for item,valid in zip(validation_reference,validation_valid)]; af_prob=[item[...,0][valid] for item,valid in zip(validation_probability,validation_valid)]
        best_threshold,best=_best_af_threshold(af_ref,af_prob)
        best_metrics,_=_full_validation(validation_reference,validation_probability,validation_valid,task,threshold=best_threshold)
        multipliers=[1.0]
    else:
        baseline=np.ones(4,dtype=np.float32); best_metrics,best=_full_validation(validation_reference,validation_probability,validation_valid,task,multipliers=baseline); best_multiplier=baseline
        for values in itertools.product((.7,1.,1.3), repeat=3):
            candidate=np.asarray((1.,)+values,dtype=np.float32); metrics,score=_full_validation(validation_reference,validation_probability,validation_valid,task,multipliers=candidate)
            if score>best: best,best_multiplier,best_metrics=score,candidate,metrics
        best_threshold=None; multipliers=best_multiplier.astype(float).tolist()
    meta={"format_version":1,"task":task,"channels":channels,"feature_names":names,"threshold":best_threshold,"class_multipliers":multipliers,"best_iteration":int(booster.best_iteration),"selection_score":float(best),"full_validation":best_metrics,"sample":sample_info,"train_records":len(train_rows),"val_records":len(val_rows),"seed":seed,"num_threads":num_threads,"parameters":params}
    (output / "metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="Train compact LightGBM pixel baselines from competition NPZ chips")
    parser.add_argument("--records",required=True); parser.add_argument("--output",required=True); parser.add_argument("--task",choices=("af","bs","all"),default="all")
    parser.add_argument("--max-pixels",type=int,default=800_000); parser.add_argument("--val-pixels",type=int,default=200_000); parser.add_argument("--num-threads",type=int,default=2); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--rounds",type=int,default=1000)
    args=parser.parse_args(argv); tasks=("af","bs") if args.task=="all" else (args.task,); result=[]
    for task in tasks: result.append(train_tree_task(load_records(args.records,task),Path(args.output)/task,max_pixels=args.max_pixels,val_pixels=args.val_pixels,num_threads=args.num_threads,seed=args.seed,rounds=args.rounds))
    print(json.dumps(result,indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
