"""Levelling every column of a strip to the strip's own median spectrum."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from building.preprocessing.crism.models.mask import Mask

FILL = 0.0


def flat_fielded_mask(cube: np.ndarray, mask: Mask) -> Mask:
    """Scale each column by the strip's median over the column's, band by band.

    Args:
        cube: The values as lines by samples by bands, levelled in place.
        mask: What the cleaning refused, kept out of every median.

    Returns:
        mask: The same mask, every refused pixel now holding the fill.
    """
    refused = mask.pixels
    strip = np.median(cube[~refused], axis=0)
    for at in range(cube.shape[1]):
        live = ~refused[:, at]
        # A column with no measurement has nothing to level, and is refused anyway.
        if live.any():
            column = cube[:, at, :]
            held = column[live]
            column[live] = held * (strip / np.median(held, axis=0))
    cube[refused] = FILL
    return replace(mask, fill=FILL)
