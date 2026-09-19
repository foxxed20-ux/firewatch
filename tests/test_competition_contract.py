import numpy as np
import pytest

from competition.metrics import (
    competition_score,
    micro_f1_af,
    micro_iou_burn,
    micro_iou_severity,
)
from competition.rle import decode_bs_rles, decode_rle, encode_bs_rles, encode_rle


def test_af_is_micro_aggregated_across_different_chip_shapes() -> None:
    reference = (
        np.array([[1, 0]], dtype=np.uint8),
        np.array([[1, 1], [1, 1]], dtype=np.uint8),
    )
    prediction = (
        np.array([[1, 1]], dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
    )

    # Global counts are TP=1, FP=1, FN=4, hence 2/(2+1+4).
    assert micro_f1_af(reference, prediction) == pytest.approx(2 / 7)


def test_empty_masks_score_one_and_missing_severity_uses_official_rule() -> None:
    empty = np.zeros((2, 3), dtype=np.uint8)
    result = competition_score(empty, empty, empty, empty)

    assert result == {
        "F1_af": 1.0,
        "IoU_burn": 1.0,
        "IoU_sev": {1: 1.0, 2: 1.0, 3: 1.0},
        "mIoU_sev": 1.0,
        "Score": 1.0,
    }

    reference = np.array([[1, 2, 0]], dtype=np.uint8)
    prediction = np.array([[1, 0, 0]], dtype=np.uint8)
    assert micro_iou_severity(reference, prediction) == {1: 1.0, 2: 0.0, 3: 1.0}


def test_burn_collapses_all_severities_but_severity_remains_exclusive() -> None:
    reference = np.array([[1, 2, 3, 0]], dtype=np.uint8)
    prediction = np.array([[3, 2, 1, 0]], dtype=np.uint8)

    assert micro_iou_burn(reference, prediction) == 1.0
    assert micro_iou_severity(reference, prediction) == {
        1: pytest.approx(0.0),
        2: pytest.approx(1.0),
        3: pytest.approx(0.0),
    }


def test_weighted_score_uses_all_official_components() -> None:
    af_reference = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    af_prediction = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    bs_reference = np.array([[1, 2, 3, 0]], dtype=np.uint8)
    bs_prediction = np.array([[1, 0, 2, 3]], dtype=np.uint8)

    result = competition_score(
        af_reference,
        af_prediction,
        bs_reference,
        bs_prediction,
    )

    assert result["F1_af"] == pytest.approx(0.5)
    assert result["IoU_burn"] == pytest.approx(0.5)
    assert result["IoU_sev"] == {
        1: pytest.approx(1.0),
        2: pytest.approx(0.0),
        3: pytest.approx(0.0),
    }
    assert result["mIoU_sev"] == pytest.approx(1 / 3)
    assert result["Score"] == pytest.approx(0.45)


def test_every_pixel_is_scored_without_a_cloud_exclusion_path() -> None:
    # Even if a caller considers this pixel cloudy, the official metric counts it.
    reference = np.array([[1]], dtype=np.uint8)
    prediction = np.array([[0]], dtype=np.uint8)
    assert micro_f1_af(reference, prediction) == 0.0


def test_non_square_rle_is_one_based_and_c_order() -> None:
    mask = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)

    encoded = encode_rle(mask)

    assert encoded == "2 2 6 1"
    np.testing.assert_array_equal(decode_rle(encoded, (2, 3)), mask)


def test_empty_rle_is_an_empty_string_and_round_trips() -> None:
    mask = np.zeros((2, 3), dtype=np.uint8)
    assert encode_rle(mask) == ""
    np.testing.assert_array_equal(decode_rle("", mask.shape), mask)


def test_bs_rles_round_trip_exclusive_class_map() -> None:
    class_map = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.uint8)
    rles = encode_bs_rles(class_map)

    assert rles == {1: "2 1 6 1", 2: "3 1 5 1", 3: "4 1"}
    np.testing.assert_array_equal(decode_bs_rles(rles, class_map.shape), class_map)


def test_overlapping_bs_rles_are_rejected() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        decode_bs_rles({1: "2 1", 2: "2 1", 3: ""}, (1, 3))


def test_invalid_competition_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        micro_f1_af(
            np.array([[1]], dtype=np.uint8),
            np.array([[255]], dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="outside"):
        micro_iou_burn(
            np.array([[0]], dtype=np.uint8),
            np.array([[255]], dtype=np.uint8),
        )
