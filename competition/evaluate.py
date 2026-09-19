"""Validation-only evaluator for CNN, LightGBM, dNBR, and probability ensembles."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .metrics import competition_score, micro_f1_af, micro_iou_burn, micro_iou_severity
from .tree import _best_af_threshold, _lgb, _predict_proba, _read_record, load_records

WEIGHTS = (
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
)  # CNN probability weight; 0/1 are individual models.


def _cnn(directory: Path, task: str):
    import torch
    from .models import build_model

    ckpt = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    cfg = ckpt["model"]
    if cfg["task"] != task:
        raise ValueError(f"{directory} is not a {task} checkpoint")
    model = build_model(cfg["in_channels"], task, cfg["base_channels"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), ckpt, device


def _cnn_probability(model, ckpt, device, x: np.ndarray, task: str) -> np.ndarray:
    import torch

    mean = np.asarray(ckpt["normalizer"]["mean"], np.float32)[:, None, None]
    std = np.asarray(ckpt["normalizer"]["std"], np.float32)[:, None, None]
    z = (np.where(np.isfinite(x), x, mean) - mean) / std
    with torch.inference_mode():
        logits = model(torch.from_numpy(z[None]).to(device)).float()
        p = (logits.sigmoid() if task == "af" else logits.softmax(1))[0].cpu().numpy()
    return np.moveaxis(p, 0, -1) if task == "bs" else p[0]


def _load_probability(records, task, cnn_dir: Path | None, tree_dir: Path | None):
    cnn = tree = None
    if cnn_dir:
        cnn = _cnn(cnn_dir, task)
    if tree_dir:
        tree = (
            _lgb().Booster(model_file=str(tree_dir / "model.txt")),
            json.loads((tree_dir / "metadata.json").read_text()),
        )
    reference = []
    valid = []
    cnn_p = []
    tree_p = []
    dnbr = []
    for r in records:
        x, y, v = _read_record(r)
        reference.append(np.where(v, y, 0).astype(np.uint8))
        valid.append(v)
        if cnn:
            model, ckpt, device = cnn
            if list(r.feature_names) != list(ckpt["feature_meta"]["names"]):
                raise ValueError("CNN feature order mismatch")
            cnn_p.append(_cnn_probability(model, ckpt, device, x, task))
        if tree:
            booster, meta = tree
            if list(r.feature_names) != list(meta["feature_names"]):
                raise ValueError("tree feature order mismatch")
            # Runtime converts LightGBM output to float32 before thresholds.
            tree_p.append(
                _predict_proba(booster, x, 65536).astype(np.float32, copy=False)
            )
        if task == "bs":
            try:
                dnbr.append(x[list(r.feature_names).index("delta_NBR")])
            except ValueError:
                raise ValueError("BS features lack delta_NBR")
    return reference, valid, cnn_p, tree_p, dnbr


def _af_metrics(reference, valid, probability):
    refs = [a[v] for a, v in zip(reference, valid)]
    probs = [a[v] for a, v in zip(probability, valid)]
    threshold, f1 = _best_af_threshold(refs, probs)
    pred = [
        np.where(v, p >= threshold, 0).astype(np.uint8)
        for p, v in zip(probability, valid)
    ]
    return {
        "F1_af": float(micro_f1_af(reference, pred)),
        "threshold": float(threshold),
    }, float(f1)


def _bs_metrics(reference, valid, probability):
    best = -1.0
    best_metrics = {}
    best_multipliers = None
    for values in __import__("itertools").product((0.7, 1.0, 1.3), repeat=3):
        multi = np.asarray((1.0,) + values, np.float32)
        pred = [
            np.where(v, (p * multi).argmax(-1), 0).astype(np.uint8)
            for p, v in zip(probability, valid)
        ]
        burn = float(micro_iou_burn(reference, pred))
        sev = micro_iou_severity(reference, pred)
        mean = float(sum(sev.values()) / 3)
        value = 0.35 * burn + 0.30 * mean
        if value > best:
            best = value
            best_multipliers = multi
            best_metrics = {
                "IoU_burn": burn,
                "mIoU_sev": mean,
                **{f"IoU_{c}": float(sev[c]) for c in (1, 2, 3)},
            }
    return {
        **best_metrics,
        "class_multipliers": best_multipliers.astype(float).tolist(),
    }, float(best)


def _score_confusion(confusion):
    burn_tp = confusion[1:, 1:].sum()
    burn_fp = confusion[0, 1:].sum()
    burn_fn = confusion[1:, 0].sum()
    burn = (
        1.0
        if burn_tp + burn_fp + burn_fn == 0
        else float(burn_tp / (burn_tp + burn_fp + burn_fn))
    )
    severity = {}
    for cls in (1, 2, 3):
        tp = confusion[cls, cls]
        union = confusion[cls, :].sum() + confusion[:, cls].sum() - tp
        severity[cls] = 1.0 if union == 0 else float(tp / union)
    mean = float(sum(severity.values()) / 3)
    return {
        "IoU_burn": burn,
        "mIoU_sev": mean,
        **{f"IoU_{c}": severity[c] for c in severity},
    }, 0.35 * burn + 0.30 * mean


def _dnbr_metrics(reference, valid, scores):
    """Four-class dNBR comparator via a 4×12 histogram, never a final recipe."""
    grid = np.asarray(
        (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), np.float32
    )
    histogram = np.zeros((4, len(grid) + 1), np.int64)
    for truth, permitted, score in zip(reference, valid, scores):
        bins = np.digitize(
            score[permitted].astype(np.float32, copy=False), grid, right=False
        )
        np.add.at(histogram, (truth[permitted], bins), 1)
    best = -1.0
    best_metrics = {}
    best_thresholds = None
    for first in range(len(grid) - 2):
        for second in range(first + 1, len(grid) - 1):
            for third in range(second + 1, len(grid)):
                confusion = np.zeros((4, 4), np.int64)
                for actual in range(4):
                    confusion[actual, 0] = histogram[actual, : first + 1].sum()
                    confusion[actual, 1] = histogram[
                        actual, first + 1 : second + 1
                    ].sum()
                    confusion[actual, 2] = histogram[
                        actual, second + 1 : third + 1
                    ].sum()
                    confusion[actual, 3] = histogram[actual, third + 1 :].sum()
                metrics, value = _score_confusion(confusion)
                if value > best:
                    best, best_metrics, best_thresholds = (
                        value,
                        metrics,
                        [float(grid[first]), float(grid[second]), float(grid[third])],
                    )
    return {
        **best_metrics,
        "thresholds": best_thresholds,
        "histogram": histogram.tolist(),
    }, float(best)


def evaluate_task(records, task, cnn_dir=None, tree_dir=None, cache_dir=None):
    reference, valid, cnn_p, tree_p, dnbr = _load_probability(
        records, task, cnn_dir, tree_dir
    )
    candidates = {}
    if task == "af":
        if cnn_p:
            candidates["cnn"] = _af_metrics(reference, valid, cnn_p)
        if tree_p:
            candidates["tree"] = _af_metrics(
                reference, valid, [p[..., 0] for p in tree_p]
            )
        if cnn_p and tree_p:
            for weight in WEIGHTS:
                candidates[f"ensemble_cnn_{weight:g}"] = _af_metrics(
                    reference,
                    valid,
                    [
                        weight * c + (1 - weight) * t[..., 0]
                        for c, t in zip(cnn_p, tree_p)
                    ],
                )
        if not candidates:
            raise ValueError(
                "AF evaluation requires at least one trained CNN or tree model"
            )
    else:
        baselines = {"dnbr_4class": _dnbr_metrics(reference, valid, dnbr)}
        if cnn_p:
            candidates["cnn"] = _bs_metrics(reference, valid, cnn_p)
        if tree_p:
            candidates["tree"] = _bs_metrics(reference, valid, tree_p)
        if cnn_p and tree_p:
            for weight in WEIGHTS:
                candidates[f"ensemble_cnn_{weight:g}"] = _bs_metrics(
                    reference,
                    valid,
                    [weight * c + (1 - weight) * t for c, t in zip(cnn_p, tree_p)],
                )
        if not candidates:
            raise ValueError(
                "BS evaluation requires at least one trained CNN or tree model"
            )
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "reference": np.stack(reference),
            "valid": np.stack(valid),
            "chip_ids": np.asarray([record.id for record in records]),
        }
        if cnn_p:
            payload["cnn_probability"] = np.stack(cnn_p).astype(np.float32)
        if tree_p:
            payload["tree_probability"] = np.stack(tree_p).astype(np.float32)
        np.savez_compressed(
            cache_dir / f"{task}_validation_probabilities.npz", **payload
        )
    name, (metrics, score) = max(candidates.items(), key=lambda item: item[1][1])
    backend = "ensemble" if name.startswith("ensemble") else name
    recipe = {
        "task": task,
        "selected": name,
        "backend": backend,
        "selection_score": score,
        **metrics,
    }
    result = {
        "recipe": recipe,
        "candidates": {
            key: {"metrics": m, "selection_score": s}
            for key, (m, s) in candidates.items()
        },
        "validation_chips": len(records),
    }
    if task == "bs":
        result["baselines"] = {
            key: {"metrics": m, "selection_score": s}
            for key, (m, s) in baselines.items()
        }
    return result


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Validation-only model and ensemble evaluation; never accepts test data"
    )
    p.add_argument("--records", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cnn-root")
    p.add_argument("--tree-root")
    p.add_argument("--cache-dir")
    a = p.parse_args(argv)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {"scope": "validation only; not an independent test score", "tasks": {}}
    for task in ("af", "bs"):
        rows = [r for r in load_records(a.records, task) if r.split == "val"]
        if not rows:
            raise ValueError(f"no validation records for {task}")
        report["tasks"][task] = evaluate_task(
            rows,
            task,
            Path(a.cnn_root) / task if a.cnn_root else None,
            Path(a.tree_root) / task if a.tree_root else None,
            a.cache_dir,
        )
    # This is a validation estimate only: each selected recipe was chosen on
    # this same split, so it must never be presented as a held-out test score.
    report["combined_validation_score"] = (
        0.35 * report["tasks"]["af"]["recipe"]["F1_af"]
        + report["tasks"]["bs"]["recipe"]["selection_score"]
    )
    (out / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out / "selected_recipes.json").write_text(
        json.dumps({k: v["recipe"] for k, v in report["tasks"].items()}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
