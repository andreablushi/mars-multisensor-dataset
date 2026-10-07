"""Levelling the columns of a strip, then removing spikes from its spectra."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from building.preprocessing.crism.models.observation import Mask

FILL = 0.0

PASSES = 3

WINDOW = 3

SIGMA = 20.0


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


def remove_spikes(cube: np.ndarray, refused: np.ndarray) -> None:
    """Remove spikes in repeated passes over a three band window, as crism_ml does.

    Args:
        cube: The flat-fielded values as lines by samples by bands, changed in place.
        refused: Lines by samples, True where the pixel is not a measurement.
    """
    if refused.all():
        return
    # A refused pixel is flat, so leaving it in would pull the threshold down.
    live = ~refused
    # The median, the distance from it and what that catches, refilled each pass.
    median = np.empty_like(cube)
    apart = np.empty_like(cube)
    caught = np.empty(cube.shape, dtype=bool)
    for _ in range(PASSES):
        moving_median(cube, WINDOW, median)
        np.subtract(median, cube, out=apart)
        np.abs(apart, out=apart)
        # crism_ml judges every sample against the measured cube's own spread.
        limit = np.mean(apart.mean(axis=-1), where=live) + SIGMA * np.mean(
            apart.std(ddof=1, axis=-1), where=live
        )
        np.greater(apart, limit, out=caught)
        np.copyto(cube, median, where=caught)


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
