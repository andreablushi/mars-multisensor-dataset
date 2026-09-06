"""crism_ml's spike removal, run on the ratioed spectra."""

from __future__ import annotations

import numpy as np

from building.preprocessing.crism import configs
from building.preprocessing.crism.correction import bands_calibration
from building.preprocessing.crism.correction.destripe import medfilt1


def remove_spikes(pixspec: np.ndarray, centre: np.ndarray) -> None:
    """Remove spikes with narrowing windows, as crism_ml does.

    Args:
        pixspec: The ratioed values as lines by samples by bands, changed in
            place.
        centre: The centre wavelength of every band it holds.

    Returns:
        None.
    """
    # The median, the distance from it and what that catches, refilled each pass.
    pixmed = np.empty_like(pixspec)
    apart = np.empty_like(pixspec)
    caught = np.empty(pixspec.shape, dtype=bool)
    for width, sigma in configs.SPIKE_PASSES:
        medfilt1(pixspec, bands_calibration.window(centre, width), out=pixmed)
        np.subtract(pixmed, pixspec, out=apart)
        np.abs(apart, out=apart)
        # crism_ml judges every sample against the whole cube's own spread.
        limit = np.mean(apart.mean(axis=-1), keepdims=True) + sigma * np.mean(
            apart.std(ddof=1, axis=-1), keepdims=True
        )
        np.greater(apart, limit, out=caught)
        np.copyto(pixspec, pixmed, where=caught)
