"""Official FireWatch competition metrics.

The competition evaluates pixels globally (micro averaging).  Inputs may be a
single NumPy array or a sequence of chip arrays; chips in a sequence may have
different spatial shapes, but each reference/prediction pair must match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


MaskBatch = NDArray[np.generic] | Sequence[NDArray[np.generic]]


@dataclass(frozen=True)
class ConfusionCounts:
    """Micro-aggregated binary confusion counts."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        if not isinstance(other, ConfusionCounts):
            return NotImplemented
        return ConfusionCounts(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )


def _chip_pairs(
    reference: MaskBatch,
    prediction: MaskBatch,
) -> list[tuple[NDArray[np.generic], NDArray[np.generic]]]:
    if isinstance(reference, np.ndarray) and isinstance(prediction, np.ndarray):
        pairs = [(reference, prediction)]
    elif isinstance(reference, np.ndarray) or isinstance(prediction, np.ndarray):
        raise TypeError(
            "reference and prediction must both be ndarrays or both be sequences"
        )
    else:
        try:
            references = list(reference)
            predictions = list(prediction)
        except TypeError as exc:
            raise TypeError(
                "reference and prediction must be ndarrays or sequences of ndarrays"
            ) from exc
        if len(references) != len(predictions):
            raise ValueError(
                "reference and prediction must contain the same number of chips"
            )
        if not references:
            raise ValueError("at least one chip is required")
        pairs = []
        for index, (ref, pred) in enumerate(zip(references, predictions)):
            if not isinstance(ref, np.ndarray) or not isinstance(pred, np.ndarray):
                raise TypeError(f"chip {index} must be a NumPy ndarray")
            pairs.append((ref, pred))

    for index, (ref, pred) in enumerate(pairs):
        if ref.shape != pred.shape:
            raise ValueError(
                f"chip {index} shape mismatch: reference {ref.shape}, "
                f"prediction {pred.shape}"
            )
        if ref.size == 0:
            raise ValueError(f"chip {index} must not be empty")
    return pairs


def _validate_classes(
    array: ArrayLike,
    allowed: tuple[int, ...],
    *,
    name: str,
) -> None:
    values = np.asarray(array)
    if not (np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_):
        raise TypeError(f"{name} must contain numeric class labels")
    if not np.all(np.isin(values, allowed)):
        present = np.unique(values)
        raise ValueError(
            f"{name} contains labels outside {allowed}: {present.tolist()}"
        )


def _binary_counts(
    pairs: list[tuple[NDArray[np.generic], NDArray[np.generic]]],
    positive: Callable[[NDArray[np.generic]], NDArray[np.bool_]],
) -> ConfusionCounts:
    total = ConfusionCounts()
    for reference, prediction in pairs:
        ref_positive = positive(reference)
        pred_positive = positive(prediction)
        total += ConfusionCounts(
            tp=int(np.count_nonzero(ref_positive & pred_positive)),
            fp=int(np.count_nonzero(~ref_positive & pred_positive)),
            fn=int(np.count_nonzero(ref_positive & ~pred_positive)),
        )
    return total


def f1_from_counts(counts: ConfusionCounts) -> float:
    """Return binary F1, treating absence in both inputs as a perfect score."""

    denominator = 2 * counts.tp + counts.fp + counts.fn
    return 1.0 if denominator == 0 else (2.0 * counts.tp) / denominator


def iou_from_counts(counts: ConfusionCounts) -> float:
    """Return binary IoU, treating absence in both inputs as a perfect score."""

    denominator = counts.tp + counts.fp + counts.fn
    return 1.0 if denominator == 0 else counts.tp / denominator


def micro_f1_af(reference: MaskBatch, prediction: MaskBatch) -> float:
    """Compute official pixel-level micro F1 for AF labels 0/1."""

    pairs = _chip_pairs(reference, prediction)
    for index, (ref, pred) in enumerate(pairs):
        _validate_classes(ref, (0, 1), name=f"AF reference chip {index}")
        _validate_classes(pred, (0, 1), name=f"AF prediction chip {index}")
    return f1_from_counts(_binary_counts(pairs, lambda values: values == 1))


def _validated_bs_pairs(
    reference: MaskBatch,
    prediction: MaskBatch,
) -> list[tuple[NDArray[np.generic], NDArray[np.generic]]]:
    pairs = _chip_pairs(reference, prediction)
    for index, (ref, pred) in enumerate(pairs):
        _validate_classes(ref, (0, 1, 2, 3), name=f"BS reference chip {index}")
        _validate_classes(pred, (0, 1, 2, 3), name=f"BS prediction chip {index}")
    return pairs


def micro_iou_burn(reference: MaskBatch, prediction: MaskBatch) -> float:
    """Compute official pixel-level micro IoU for binary burn (BS > 0)."""

    pairs = _validated_bs_pairs(reference, prediction)
    return iou_from_counts(_binary_counts(pairs, lambda values: values > 0))


def micro_iou_severity(
    reference: MaskBatch,
    prediction: MaskBatch,
) -> dict[int, float]:
    """Return official micro IoU for BS severity classes 1, 2 and 3."""

    pairs = _validated_bs_pairs(reference, prediction)
    return {
        severity: iou_from_counts(
            _binary_counts(pairs, lambda values, cls=severity: values == cls)
        )
        for severity in (1, 2, 3)
    }


def competition_score(
    af_reference: MaskBatch,
    af_prediction: MaskBatch,
    bs_reference: MaskBatch,
    bs_prediction: MaskBatch,
) -> dict[str, object]:
    """Compute all official FireWatch metrics and their weighted score.

    There is deliberately no cloud or validity mask argument: every submitted
    competition pixel, including cloudy pixels, participates in the metric.
    """

    f1_af = micro_f1_af(af_reference, af_prediction)
    iou_burn = micro_iou_burn(bs_reference, bs_prediction)
    iou_sev = micro_iou_severity(bs_reference, bs_prediction)
    miou_sev = sum(iou_sev.values()) / 3.0
    score = 0.35 * f1_af + 0.35 * iou_burn + 0.30 * miou_sev
    return {
        "F1_af": f1_af,
        "IoU_burn": iou_burn,
        "IoU_sev": iou_sev,
        "mIoU_sev": miou_sev,
        "Score": score,
    }
