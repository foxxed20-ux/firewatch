import torch
import numpy as np

from competition.models import build_model
from competition.train import _best_af_threshold, _binary_loss


def test_af_shape_is_binary_for_competition_chip():
    model = build_model(9, "af", base_channels=8)
    assert model(torch.randn(2, 9, 256, 256)).shape == (2, 1, 256, 256)


def test_bs_shape_is_four_classes_for_competition_chip():
    model = build_model(17, "bs", base_channels=8)
    assert model(torch.randn(1, 17, 512, 512)).shape == (1, 4, 512, 512)


def test_af_threshold_search_can_select_below_old_fixed_grid():
    probability = np.full((2, 2), .10, dtype=np.float32)
    threshold, f1 = _best_af_threshold([np.ones((2, 2), dtype=np.uint8)], [probability])
    assert threshold <= float(probability[0, 0])
    assert f1 == 1.0


def test_binary_loss_and_gradient_are_finite_with_ignore_255():
    logits = torch.tensor([[[[0.2, -0.7], [1.1, 0.0]]]], requires_grad=True)
    target = torch.tensor([[[1, 255], [0, 255]]])
    valid = target != 255
    loss = _binary_loss(logits, target, valid, pos_weight=3.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
