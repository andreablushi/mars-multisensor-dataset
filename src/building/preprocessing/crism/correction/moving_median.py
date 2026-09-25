"""A moving median along the bands, and how many bands its window covers."""

from __future__ import annotations

import numpy as np


def window_size(centre: np.ndarray, width: float) -> int:
    """Return how many bands a window of a given width covers.

    Args:
        centre: The centre wavelength of every kept band, in order.
        width: How far the window should reach, in nm.

    Returns:
        size: An odd band count whose span fits inside the width, never below three.
    """
    if centre.size < 2:
        return 3
    step = np.abs(np.diff(centre)).mean()
    size = int(width // step) + 1
    return max(size - 1 + size % 2, 3)


def moving_median(
    array: np.ndarray, size: int, out: np.ndarray | None = None
) -> np.ndarray:
    """Return a moving median along the last axis, truncated at the ends.

    Args:
        array: The values to filter.
        size: How many samples wide the window is.
        out: The array to fill, or None to allocate one.

    Returns:
        values: The filtered values, the same shape as the input.
    """
    left, right = size // 2, size - size // 2
    if out is None:
        out = np.empty_like(array)
    for at in range(array.shape[-1]):
        window = array[..., max(at - left, 0) : at + right]
        held = window.shape[-1]
        # Partitioned rather than sorted, which stops at the middle and writes in place.
        part = np.partition(window, held // 2, axis=-1)
        if held % 2:
            out[..., at] = part[..., held // 2]
        else:
            out[..., at] = 0.5 * (part[..., held // 2 - 1] + part[..., held // 2])
    return out
