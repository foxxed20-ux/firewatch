import numpy as np

from competition.metrics import micro_iou_burn, micro_iou_severity
from ops.tune_postprocess import confusion_from_cache, score_cache, tune


def test_confusion_score_matches_official_micro_metrics():
    reference = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
    valid = np.ones_like(reference, dtype=bool)
    probability = np.eye(4, dtype=np.float32)[reference]
    matrix, metrics, _ = score_cache(
        reference, valid, probability, np.ones(4, dtype=np.float32), 2
    )
    prediction = probability.argmax(-1).astype(np.uint8)
    assert matrix.tolist() == np.eye(4, dtype=int).tolist()
    assert metrics["IoU_burn"] == micro_iou_burn(reference, prediction)
    assert (
        metrics["mIoU_sev"]
        == sum(micro_iou_severity(reference, prediction).values()) / 3
    )


def test_coordinate_tuning_never_decreases_score():
    reference = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
    valid = np.ones_like(reference, dtype=bool)
    probability = np.eye(4, dtype=np.float32)[reference]
    report = tune(reference, valid, probability, [1.0, 1.0, 1.0, 1.0], 2)
    assert report["final"]["score"] >= report["initial"]["score"]
    assert report["tried_weights"] > 0
