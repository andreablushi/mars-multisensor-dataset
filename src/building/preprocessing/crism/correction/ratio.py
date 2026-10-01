"""Dividing each spectrum by a bland one from its own column."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from building.preprocessing.crism.models.mask import Mask

# What the ratio writes where a pixel is not a measurement, in its own units.
FILL = 0.0


def ratio_by_column_median(cube: np.ndarray, mask: Mask) -> Mask:
    """Use the median of a column for ratioing, as crism_ml's ColMed does.

    Args:
        cube: The values as lines by samples by bands, divided in place.
        mask: What the cleaning refused, kept out of the median.

    Returns:
        mask: The same mask, in the units the ratio leaves the cube in.
    """
    divide_by_column_median(cube, ~mask.pixels)
    cube[mask.pixels] = FILL
    return replace(mask, fill=FILL)


def divide_by_column_median(cube: np.ndarray, valid: np.ndarray) -> None:
    """Divide the valid pixels of every column by their own median spectrum.

    Args:
        cube: The values as lines by samples by bands, divided in place.
        valid: Lines by samples, True where the pixel is a measurement.
    """
    for at in range(cube.shape[1]):
        live = valid[:, at]
        # A column with no measurement has nothing to ratio, and is refused anyway.
        if live.any():
            column = cube[:, at, :]
            held = column[live]
            column[live] = held / np.median(held, axis=0)
