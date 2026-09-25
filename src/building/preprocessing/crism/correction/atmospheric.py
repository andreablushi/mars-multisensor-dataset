"""Dropping the bands where the Martian atmosphere absorbs, not the surface."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from building.configs.crism import ATMOSPHERIC, Detector
from building.preprocessing.crism.models.mask import Mask


def remove_atmospheric_bands(
    cube: np.ndarray, mask: Mask, centre: np.ndarray, detector: Detector
) -> Mask:
    """Drop the bands whose depth the atmosphere sets rather than the ground.

    Args:
        cube: The masked values as lines by samples by bands, filled in place.
        mask: What that masking refused.
        centre: The centre wavelength of every band.
        detector: Which detector, `l` or `s`, which picks the windows.

    Returns:
        mask: The mask with those bands recorded.
    """
    caught = np.zeros(centre.shape, dtype=bool)
    for low, high in ATMOSPHERIC[detector]:
        caught |= (centre >= low) & (centre <= high)

    # Only what masking still counted as usable is being taken away.
    caught &= ~mask.bands
    cube[:, :, caught] = mask.fill
    return replace(mask, bands=mask.bands | caught)
