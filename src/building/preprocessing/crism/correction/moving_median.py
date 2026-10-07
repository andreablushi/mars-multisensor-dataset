"""A moving median along the bands."""

from __future__ import annotations

import numpy as np


def moving_median(array: np.ndarray, size: int, out: np.ndarray) -> None:
    """Fill an array with a moving median along the last axis, truncated at the ends.

    Args:
        array: The values to filter.
        size: How many samples wide the window is.
        out: The array to fill, the same shape as the values.
    """
    left, right = size // 2, size - size // 2
    for at in range(array.shape[-1]):
        window = array[..., max(at - left, 0) : at + right]
        held = window.shape[-1]
        # Partitioned rather than sorted, which stops at the middle and writes in place.
        part = np.partition(window, held // 2, axis=-1)
        if held % 2:
            out[..., at] = part[..., held // 2]
        else:
            out[..., at] = 0.5 * (part[..., held // 2 - 1] + part[..., held // 2])
