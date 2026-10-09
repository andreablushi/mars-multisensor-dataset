"""Dividing each pixel by the cosine of its incidence, which leaves the albedo."""

from __future__ import annotations

import numpy as np

from building.configs.crism import FILL

# The incidence past which the sun is below the horizon, in degrees.
HORIZON = 90.0


def photometric_valid(
    cube: np.ndarray, incidence: np.ndarray, valid: np.ndarray, bands: np.ndarray
) -> np.ndarray:
    """Divide every measured pixel by the cosine of its incidence, filling the unlit.

    Args:
        cube: The flat-fielded I/F as lines by samples by bands, changed in place.
        incidence: Lines by samples, the sun's incidence in degrees.
        valid: Lines by samples, True where the pixel is a measurement.
        bands: One flag per band, True where the band is in play.

    Returns:
        valid: Lines by samples, True where the pixel is still a measurement.
    """
    # A backplane fill is no angle, so only an incidence above the horizon is read.
    lit = (incidence >= 0.0) & (incidence < HORIZON)
    kept = valid & lit
    cosine = np.cos(np.radians(np.where(lit, incidence, 0.0)))
    np.divide(cube, cosine[:, :, None], out=cube, where=kept[:, :, None] & bands)
    np.copyto(cube, FILL, where=(valid & ~lit)[:, :, None] & bands)
    return kept
