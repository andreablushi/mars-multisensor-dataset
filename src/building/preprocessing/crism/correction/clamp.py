"""Refusing the pixels whose ratioed spectrum leaves what a ratio can hold."""

from __future__ import annotations

import numpy as np

from building.preprocessing.crism.correction.ratio import FILL

RATIO = (-0.5, 3.0)


def bounded_valid(cube: np.ndarray, valid: np.ndarray, bands: np.ndarray) -> np.ndarray:
    """Fill every measured pixel with a band in play outside the ratio's range.

    Args:
        cube: The ratioed values as lines by samples by bands, filled in place.
        valid: Lines by samples, True where the pixel is a measurement.
        bands: One flag per band, True where the band is in play.

    Returns:
        valid: Lines by samples, True where the pixel is still a measurement.
    """
    low, high = RATIO
    held = cube[:, :, bands]
    outside = valid & ((held < low) | (held > high)).any(axis=2)
    np.copyto(cube, FILL, where=outside[:, :, None] & bands)
    return valid & ~outside
