"""Run-length encoding for FireWatch competition masks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _shape_tuple(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("shape must contain exactly two dimensions")
    try:
        height, width = (index(value) for value in shape)
    except TypeError as exc:
        raise TypeError("shape dimensions must be integers") from exc
    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive")
    return height, width


def _binary_mask(mask: ArrayLike, *, name: str = "mask") -> NDArray[np.bool_]:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not (np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_):
        raise TypeError(f"{name} must contain numeric values")
    if not np.all(np.isin(values, (0, 1))):
        raise ValueError(f"{name} must contain only 0 and 1")
    return values.astype(bool, copy=False)


def _bs_class_map(class_map: ArrayLike) -> NDArray[np.generic]:
    values = np.asarray(class_map)
    if values.ndim != 2:
        raise ValueError("BS class map must be a two-dimensional array")
    if values.size == 0:
        raise ValueError("BS class map must not be empty")
    if not (np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_):
        raise TypeError("BS class map must contain numeric class labels")
    if not np.all(np.isin(values, (0, 1, 2, 3))):
        raise ValueError("BS class map must contain only classes 0, 1, 2 and 3")
    return values


def encode_rle(mask: ArrayLike) -> str:
    """Encode a binary 2D mask as one-based, row-major (C-order) RLE."""

    flat = _binary_mask(mask).reshape(-1, order="C")
    padded = np.concatenate(
        (np.array([False]), flat, np.array([False]))
    ).astype(np.int8, copy=False)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1) + 1
    ends = np.flatnonzero(transitions == -1) + 1
    lengths = ends - starts
    return " ".join(
        str(value)
        for start, length in zip(starts, lengths)
        for value in (int(start), int(length))
    )


def decode_rle(rle: str, shape: Sequence[int]) -> NDArray[np.uint8]:
    """Decode canonical one-based, row-major RLE into a uint8 mask."""

    height, width = _shape_tuple(shape)
    if not isinstance(rle, str):
        raise TypeError("rle must be a string")
    tokens = rle.split()
    if len(tokens) % 2:
        raise ValueError("RLE must contain start/length pairs")
    try:
        numbers = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("RLE tokens must be integers") from exc

    flat = np.zeros(height * width, dtype=np.uint8)
    previous_end = -1
    for start_one_based, length in zip(numbers[::2], numbers[1::2]):
        if start_one_based < 1:
            raise ValueError("RLE starts are one-based and must be positive")
        if length <= 0:
            raise ValueError("RLE run lengths must be positive")
        start = start_one_based - 1
        end = start + length
        if end > flat.size:
            raise ValueError("RLE run exceeds the requested mask shape")
        if start <= previous_end:
            raise ValueError("RLE runs must be ordered, disjoint and non-adjacent")
        flat[start:end] = 1
        previous_end = end
    return flat.reshape((height, width), order="C")


def encode_bs_rles(class_map: ArrayLike) -> dict[int, str]:
    """Encode exclusive BS classes 1, 2 and 3 from a single class map."""

    values = _bs_class_map(class_map)
    return {severity: encode_rle(values == severity) for severity in (1, 2, 3)}


def decode_bs_rles(
    rles: Mapping[int, str],
    shape: Sequence[int],
) -> NDArray[np.uint8]:
    """Decode severity RLEs, rejecting missing classes and pixel overlap."""

    if set(rles) != {1, 2, 3}:
        raise ValueError("BS RLE mapping must have exactly integer keys 1, 2 and 3")
    result = np.zeros(_shape_tuple(shape), dtype=np.uint8)
    for severity in (1, 2, 3):
        mask = decode_rle(rles[severity], result.shape).astype(bool, copy=False)
        if np.any((result != 0) & mask):
            raise ValueError("BS severity RLE masks must be mutually exclusive")
        result[mask] = severity
    return result
