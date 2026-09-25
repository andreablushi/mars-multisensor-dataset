"""crism_ml's spike removal, run on the ratioed spectra."""

from __future__ import annotations

import numpy as np

from building.preprocessing.crism.correction import moving_median

# The windows crism_ml despikes with, in nm, at 20 deviations rather than five.
SPIKE_PASSES = ((72.0, 20.0), (46.0, 20.0), (20.0, 20.0))


def remove_spikes(cube: np.ndarray, centre: np.ndarray, refused: np.ndarray) -> None:
    """Remove spikes with narrowing windows, as crism_ml does.

    Args:
        cube: The ratioed values as lines by samples by bands, changed in place.
        centre: The centre wavelength of every band it holds.
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
    for width, sigma in SPIKE_PASSES:
        size = moving_median.window_size(centre, width)
        moving_median.moving_median(cube, size, out=median)
        np.subtract(median, cube, out=apart)
        np.abs(apart, out=apart)
        # crism_ml judges every sample against the measured cube's own spread.
        limit = np.mean(apart.mean(axis=-1), where=live) + sigma * np.mean(
            apart.std(ddof=1, axis=-1), where=live
        )
        np.greater(apart, limit, out=caught)
        np.copyto(cube, median, where=caught)
