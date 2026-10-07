"""crism_ml's spike removal, run on the flat-fielded spectra."""

from __future__ import annotations

import numpy as np

from building.preprocessing.crism.correction import moving_median

PASSES = 3

WINDOW = 3

SIGMA = 20.0


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
        moving_median.moving_median(cube, WINDOW, median)
        np.subtract(median, cube, out=apart)
        np.abs(apart, out=apart)
        # crism_ml judges every sample against the measured cube's own spread.
        limit = np.mean(apart.mean(axis=-1), where=live) + SIGMA * np.mean(
            apart.std(ddof=1, axis=-1), where=live
        )
        np.greater(apart, limit, out=caught)
        np.copyto(cube, median, where=caught)
