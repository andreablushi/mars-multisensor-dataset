"""Dividing each spectrum by a bland one from its own column."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from building.preprocessing.crism.models.mask import Mask

# What the ratio writes where a pixel is not a measurement, in its own units.
FILL = 0.0


def ratio_colmed(pixspec: np.ndarray, mask: Mask) -> Mask:
    """Use the median of a column for ratioing, as crism_ml's ColMed does.

    Args:
        pixspec: The values as lines by samples by bands, divided in place.
        mask: What the cleaning refused, kept out of the median.

    Returns:
        mask: The same mask, in the units the ratio leaves the cube in.
    """
    rem = mask.pixels
    for at in range(pixspec.shape[1]):
        live = ~rem[:, at]
        # A column with no measurement has nothing to ratio, and is refused anyway.
        if live.any():
            column = pixspec[:, at, :]
            held = column[live]
            column[live] = held / np.median(held, axis=0)
    pixspec[rem] = FILL
    return replace(mask, fill=FILL)
