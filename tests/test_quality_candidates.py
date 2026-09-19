from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from competition.models import build_model
from competition.train import _multiclass_loss
from ops.compare_quality_candidates import (
    _assert_same,
    _load,
    _metrics,
    _per_chip,
)
from ops.cache_quality_cnn import (
    D4,
    _inverse_transform,
    _transform,
    forward_probability,
)
from ops.train_quality_cnn import EMA, _save


def test_d4_inverse_is_exact_for_every_non_square_transform():
    source = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    variants = set()
    for turns, mirror in D4:
        transformed = _transform(source, turns, mirror)
        assert torch.equal(_inverse_transform(transformed, turns, mirror), source)
        variants.add((tuple(transformed.shape), transformed.numpy().tobytes()))
    assert len(variants) == 8


def test_d4_forward_matches_none_for_spatially_equivariant_model():
    class Pointwise(torch.nn.Module):
        def forward(self, x):
            return torch.cat((x[:, :1], -x[:, :1], x[:, :1] * 0.5, x[:, :1] * 0), 1)

    x = torch.arange(2 * 1 * 5 * 7, dtype=torch.float32).reshape(2, 1, 5, 7) / 70
    direct = forward_probability(Pointwise(), x, "bs", "none")
    augmented = forward_probability(Pointwise(), x, "bs", "d4")
    assert direct.shape == augmented.shape == (2, 4, 5, 7)
    assert direct.dtype == augmented.dtype == torch.float32
    assert direct.device.type == augmented.device.type == "cpu"
    torch.testing.assert_close(augmented, direct, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        augmented.sum(1), torch.ones((2, 5, 7)), rtol=1e-6, atol=1e-7
    )


def _one_hot_probability(prediction: np.ndarray) -> np.ndarray:
    return np.eye(4, dtype=np.float32)[prediction]


def _direct_confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 4), dtype=np.int64)
    for actual, predicted in zip(truth.reshape(-1), prediction.reshape(-1)):
        result[int(actual), int(predicted)] += 1
    return result


def _direct_bs_metrics(confusion: np.ndarray) -> tuple[float, float, list[float]]:
    burn_tp = int(confusion[1:, 1:].sum())
    burn_union = burn_tp + int(confusion[0, 1:].sum()) + int(confusion[1:, 0].sum())
    burn = 1.0 if burn_union == 0 else burn_tp / burn_union
    severity = []
    for class_id in (1, 2, 3):
        tp = int(confusion[class_id, class_id])
        union = int(confusion[class_id].sum() + confusion[:, class_id].sum() - tp)
        severity.append(1.0 if union == 0 else tp / union)
    weighted = 0.35 * burn + 0.30 * sum(severity) / 3
    return weighted, burn, severity


def test_per_chip_score_equals_direct_aggregate_confusion():
    reference = np.asarray(
        [[[0, 1, 2], [3, 1, 0]], [[3, 2, 1], [0, 0, 2]]], dtype=np.uint8
    )
    valid = np.asarray(
        [[[True, True, True], [True, False, True]], [[True, True, True], [True, True, False]]]
    )
    baseline_prediction = np.asarray(
        [[[0, 1, 0], [3, 2, 0]], [[3, 1, 1], [0, 2, 2]]], dtype=np.uint8
    )
    candidate_prediction = np.asarray(
        [[[1, 1, 2], [3, 1, 0]], [[3, 2, 0], [0, 0, 2]]], dtype=np.uint8
    )
    baseline = _per_chip(
        reference,
        valid,
        _one_hot_probability(baseline_prediction),
        _one_hot_probability(candidate_prediction),
        cnn_weight=0.0,
        multipliers=(1.0, 1.0, 1.0, 1.0),
    )
    candidate = _per_chip(
        reference,
        valid,
        _one_hot_probability(baseline_prediction),
        _one_hot_probability(candidate_prediction),
        cnn_weight=1.0,
        multipliers=(1.0, 1.0, 1.0, 1.0),
    )
    direct_baseline = sum(
        (_direct_confusion(y[keep], p[keep]) for y, p, keep in zip(reference, baseline_prediction, valid)),
        np.zeros((4, 4), np.int64),
    )
    direct_candidate = sum(
        (_direct_confusion(y[keep], p[keep]) for y, p, keep in zip(reference, candidate_prediction, valid)),
        np.zeros((4, 4), np.int64),
    )
    np.testing.assert_array_equal(baseline.sum(0), direct_baseline)
    np.testing.assert_array_equal(candidate.sum(0), direct_candidate)
    for matrix in (direct_baseline, direct_candidate):
        weighted, burn, severity = _direct_bs_metrics(matrix)
        observed = _metrics(matrix[None], fixed_af_f1=0.9)
        assert observed["weighted_bs_score"] == pytest.approx(weighted)
        assert observed["IoU_burn"] == pytest.approx(burn)
        assert observed["mIoU_severity"] == pytest.approx(sum(severity) / 3)
        np.testing.assert_allclose(
            [observed[f"IoU_{i}"] for i in (1, 2, 3)], severity
        )
        assert observed["combined_score"] == pytest.approx(0.35 * 0.9 + weighted)


def test_probability_cache_rejects_duplicate_chip_ids(tmp_path):
    reference = np.zeros((2, 2, 3), dtype=np.uint8)
    valid = np.ones_like(reference, dtype=bool)
    probability = np.full((*reference.shape, 4), 0.25, dtype=np.float32)
    path = tmp_path / "cache.npz"
    np.savez(
        path,
        reference=reference,
        valid=valid,
        cnn_probability=probability,
        chip_ids=np.asarray(["BS_duplicate", "BS_duplicate"]),
    )
    with pytest.raises(ValueError, match="chip_ids must be unique shape N"):
        _load(path, "cnn", require_ids=True)


@pytest.mark.parametrize(
    ("field", "message"),
    (("reference", "reference differs"), ("valid", "valid differs"), ("chip_ids", "chip_ids order differs")),
)
def test_assert_same_rejects_reference_valid_and_order_mismatch(tmp_path, field, message):
    canonical = {
        "path": tmp_path / "canonical.npz",
        "reference": np.zeros((2, 2, 2), dtype=np.uint8),
        "valid": np.ones((2, 2, 2), dtype=bool),
        "chip_ids": np.asarray(["BS_a", "BS_b"]),
    }
    other = {
        "path": tmp_path / "other.npz",
        "reference": canonical["reference"].copy(),
        "valid": canonical["valid"].copy(),
        "chip_ids": canonical["chip_ids"].copy(),
    }
    if field == "chip_ids":
        other[field] = other[field][::-1]
    else:
        other[field].reshape(-1)[0] = 1 if field == "reference" else False
    with pytest.raises(ValueError, match=message):
        _assert_same(canonical, other)


def test_ema_real_step_is_detached_and_checkpoint_roundtrips(tmp_path):
    torch.manual_seed(7)
    model = build_model(1, "bs", base_channels=1)
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    ema = EMA(model, decay=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(2, 1, 8, 8)
    y = torch.arange(2 * 8 * 8).reshape(2, 8, 8).remainder(4)
    valid = torch.ones_like(y, dtype=torch.bool)
    optimizer.zero_grad(set_to_none=True)
    loss = _multiclass_loss(model(x), y, valid, torch.ones(4))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    trained = {key: value.detach().clone() for key, value in model.state_dict().items()}
    assert any(not torch.equal(initial[key], trained[key]) for key in initial)

    ema.update(model)
    for key in initial:
        torch.testing.assert_close(ema.state[key], (initial[key] + trained[key]) / 2)
    first_key = next(iter(model.state_dict()))
    protected = ema.state[first_key].clone()
    with torch.no_grad():
        dict(model.named_parameters())[first_key].add_(1)
    assert torch.equal(ema.state[first_key], protected)

    ema.copy_to(model)
    source = {
        "model": {"task": "bs", "in_channels": 1, "base_channels": 1},
        "model_state": copy.deepcopy(initial),
        "normalizer": {"mean": [0.0], "std": [1.0]},
        "feature_meta": {"names": ["a"]},
    }
    path = tmp_path / "ema.pt"
    _save(path, model, source, 1, 0.25, {"IoU_burn": 0.25}, {"seed": 7})
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    restored = build_model(1, "bs", base_channels=1)
    restored.load_state_dict(loaded["model_state"], strict=True)
    assert restored(x).shape == (2, 4, 8, 8)
    assert loaded["epoch"] == 1 and loaded["quality_config"] == {"seed": 7}
