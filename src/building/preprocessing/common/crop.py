"""Cutting the arrays of one observation down to the samples a box keeps."""

from __future__ import annotations

import numpy as np


def marked(held: np.ndarray) -> np.ndarray | None:
    """Return one mask, or nothing at all where it marks every sample.

    Args:
        held: The mask over the samples a crop keeps.

    Returns:
        mask: The mask, or None where every sample is true and it says nothing the shape
            does not.
    """
    return None if held.all() else held


def taken(array: np.ndarray, bounds: tuple[np.ndarray, ...]) -> np.ndarray:
    """Return the part of one array a cut's bounds keep of its leading axes.

    Args:
        array: The array to cut, whose leading axes are the ground's.
        bounds: The samples to keep of each of those axes.

    Returns:
        held: The part that is left, every axis past the ground's kept whole.
    """
    # Neighbouring bounds are sliced rather than gathered, which costs nothing to take.
    runs = tuple(
        slice(int(held[0]), int(held[-1]) + 1)
        for held in bounds
        if held.size and np.all(np.diff(held) == 1)
    )
    if len(runs) == len(bounds):
        return array[runs]
    return array[np.ix_(*bounds)] if len(bounds) > 1 else array[bounds[0]]
