"""Dividing each pixel by its own mean, leaving its spectral shape, not brightness."""

from __future__ import annotations

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism.correction.ratio import FILL

FLOOR = 0.2

MULTISPECTRAL = np.isin(configs.BANDS_NM, configs.MULTISPECTRAL_BANDS_NM)


def shaped_valid(cube: np.ndarray, valid: np.ndarray, bands: np.ndarray) -> np.ndarray:
    """Divide every measured pixel by its multispectral mean, filling the too dim ones.

    Args:
        cube: The ratioed values as lines by samples by bands, changed in place.
        valid: Lines by samples, True where the pixel is a measurement.
        bands: One flag per band, True where the band is in play.

    Returns:
        valid: Lines by samples, True where the pixel is still a measurement.
    """
    mean = cube[:, :, bands & MULTISPECTRAL].mean(axis=2)
    dim = valid & (mean < FLOOR)
    kept = valid & ~dim
    np.divide(cube, mean[:, :, None], out=cube, where=kept[:, :, None] & bands)
    np.copyto(cube, FILL, where=dim[:, :, None] & bands)
    return kept
