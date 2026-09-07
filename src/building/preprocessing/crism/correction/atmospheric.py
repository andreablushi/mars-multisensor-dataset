"""Dropping the bands where the Martian atmosphere absorbs, not the surface."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from building.preprocessing.crism import configs
from building.preprocessing.crism.correction import bands_calibration
from building.preprocessing.crism.models.mask import Mask


def remove_atmospheric_bands(
    cube: np.ndarray, mask: Mask, table: np.ndarray, detector: str
) -> Mask:
    """Drop the bands whose depth the atmosphere sets rather than the ground.

    Args:
        cube: The values as lines by samples by bands, already masked, filled
            in place.
        mask: What that masking refused.
        table: The centre wavelength of every column and band.
        detector: Which detector, `l` or `s`, which picks the windows.

    Returns:
        mask: The mask with those bands recorded.
    """
    centre = bands_calibration.centres(table)
    caught = np.zeros(centre.shape, dtype=bool)
    for low, high in configs.ATMOSPHERIC[detector]:
        caught |= ~np.isnan(centre) & (centre >= low) & (centre <= high)

    # Only what masking still counted as usable is being taken away.
    caught &= ~mask.bands
    cube[:, :, caught] = mask.fill
    return replace(mask, bands=mask.bands | caught, atmospheric=caught)
