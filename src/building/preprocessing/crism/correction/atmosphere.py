"""Removing what the Martian atmosphere adds to a scan, rather than the ground."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from building.configs.crism import ATMOSPHERIC_BANDS_NM, CO2_BAND_NM, Detector
from building.preprocessing.crism.models.observation import Mask


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


def atmospheric_mask(
    cube: np.ndarray, mask: Mask, centres: np.ndarray, detector: Detector
) -> Mask:
    """Drop the bands whose depth the atmosphere sets rather than the ground.

    Args:
        cube: The masked values as lines by samples by bands, filled in place.
        mask: What that masking refused.
        centres: The centre wavelength of every band.
        detector: Which detector, `l` or `s`, which picks the windows.

    Returns:
        mask: The mask with those bands recorded.
    """
    caught = np.zeros(centres.shape, dtype=bool)
    for low, high in ATMOSPHERIC_BANDS_NM[detector]:
        caught |= (centres >= low) & (centres <= high)

    # Only what masking still counted as usable is being taken away.
    caught &= ~mask.bands
    cube[:, :, caught] = mask.fill
    return replace(mask, bands=mask.bands | caught)
