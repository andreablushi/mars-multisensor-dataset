"""Saying which pixels of a cube are not measurements."""

from __future__ import annotations

import numpy as np

# The range a brightness can take, its floor below zero so noise there survives.
BRIGHTNESS = (-0.05, 1.0)


def refused_pixels(cube: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Return which pixels are not a measurement.

    Args:
        cube: The values as lines by samples by bands.
        columns: One flag per sample, True where the column was never calibrated.

    Returns:
        refused: Lines by samples, True where a band is no reading or the column dead.
    """
    floor, ceiling = BRIGHTNESS
    # A brightness outside what light can do is not a reading, and neither is a NaN.
    refused = ~((cube >= floor) & (cube <= ceiling)).all(axis=2)
    # A pixel is unusable when its column is dead or any of its bands is.
    refused[:, columns] = True
    return refused
