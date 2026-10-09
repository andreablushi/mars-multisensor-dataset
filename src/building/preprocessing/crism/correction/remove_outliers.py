"""Levelling the columns of a strip, then removing spikes from its spectra."""

from __future__ import annotations

import numpy as np

from building.configs.crism import FILL


def flat_field(cube: np.ndarray, refused: np.ndarray) -> None:
    """Scale each column by the strip's median over the column's, band by band.

    Args:
        cube: The values as lines by samples by bands, levelled in place.
        refused: Lines by samples, True where the pixel is not a measurement.
    """
    strip = np.median(cube[~refused], axis=0)
    for at in range(cube.shape[1]):
        live = ~refused[:, at]
        # A column with no measurement has nothing to level, and is refused anyway.
        if live.any():
            column = cube[:, at, :]
            held = column[live]
            column[live] = held * (strip / np.median(held, axis=0))
    cube[refused] = FILL


def remove_spikes(
    cube: np.ndarray, refused: np.ndarray, passes: int, window: int, sigma: float
) -> None:
    """Remove spikes in repeated passes over a moving median, as crism_ml does.

    Args:
        cube: The flat-fielded values as lines by samples by bands, changed in place.
        refused: Lines by samples, True where the pixel is not a measurement.
        passes: How many times the cube is filtered.
        window: How many bands wide the moving median is.
        sigma: How many spreads from the median a value may sit.
    """
    # A refused pixel is flat, so leaving it in would pull the threshold down.
    live = ~refused
    # The median, the distance from it and what that catches, refilled each pass.
    median = np.empty_like(cube)
    apart = np.empty_like(cube)
    caught = np.empty(cube.shape, dtype=bool)
    for _ in range(passes):
        moving_median(cube, window, median)
        np.subtract(median, cube, out=apart)
        np.abs(apart, out=apart)
        # crism_ml judges every sample against the measured cube's own spread.
        limit = np.mean(apart.mean(axis=-1), where=live) + sigma * np.mean(
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
