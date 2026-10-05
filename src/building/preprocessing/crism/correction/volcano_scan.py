"""Dividing out the atmosphere's gas bands, scaled to each pixel's CO2 depth."""

from __future__ import annotations

import numpy as np

from building.configs.crism import CO2_BAND_NM
from building.preprocessing.crism.models.mask import Mask


def remove_atmosphere(
    cube: np.ndarray, mask: Mask, transmission: np.ndarray, centres: np.ndarray
) -> None:
    """Divide each pixel by the transmission raised to its own CO2 depth, as CAT does.

    Args:
        cube: The masked I/F as lines by samples by bands, divided in place.
        mask: What that masking refused, which neither reads a depth nor is divided.
        transmission: The atmosphere's transmission per sample and band, NaN unknown.
        centres: The centre wavelength of every band.
    """
    near, far = (int(np.nanargmin(np.abs(centres - nm))) for nm in CO2_BAND_NM)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.log(cube[:, :, near] / cube[:, :, far]) / np.log(
            transmission[:, near] / transmission[:, far]
        )
    read = ~mask.pixels & np.isfinite(depth) & (depth > 0)
    # A pixel whose depth is lost in its noise takes the strip's own.
    depth[~read] = np.median(depth[read]) if read.any() else 1.0
    divided = np.isfinite(transmission) & ~mask.bands
    np.divide(cube, transmission ** depth[:, :, None], out=cube, where=divided)
