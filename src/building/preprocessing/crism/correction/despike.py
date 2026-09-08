"""crism_ml's spike removal, run on the ratioed spectra."""

from __future__ import annotations

import numpy as np

from building.preprocessing.crism.correction import bands_calibration
from building.preprocessing.crism.correction.destripe import medfilt1

# The windows crism_ml despikes with, in nm, at 20 deviations rather than five.
SPIKE_PASSES = ((72.0, 20.0), (46.0, 20.0), (20.0, 20.0))


def remove_spikes(pixspec: np.ndarray, centre: np.ndarray) -> None:
    """Remove spikes with narrowing windows, as crism_ml does.

    Args:
        pixspec: The ratioed values as lines by samples by bands, changed in
            place.
        centre: The centre wavelength of every band it holds.
    """
    # The median, the distance from it and what that catches, refilled each pass.
    pixmed = np.empty_like(pixspec)
    apart = np.empty_like(pixspec)
    caught = np.empty(pixspec.shape, dtype=bool)
    for width, sigma in SPIKE_PASSES:
        medfilt1(pixspec, bands_calibration.window(centre, width), out=pixmed)
        np.subtract(pixmed, pixspec, out=apart)
        np.abs(apart, out=apart)
        # crism_ml judges every sample against the whole cube's own spread.
        limit = np.mean(apart.mean(axis=-1)) + sigma * np.mean(
            apart.std(ddof=1, axis=-1)
        )
        np.greater(apart, limit, out=caught)
        np.copyto(pixspec, pixmed, where=caught)
