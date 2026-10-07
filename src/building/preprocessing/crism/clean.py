"""Cleaning each CRISM detector step by step, and lighting the joined observation."""

from __future__ import annotations

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism.correction import atmosphere, remove_outliers
from building.preprocessing.crism.correction.mask import NoMeasurement, refused_mask
from building.preprocessing.crism.models.observation import DetectorCube

# The incidence past which the sun is below the horizon, in degrees.
HORIZON = 90.0


def clean_detectors(
    detectors: dict[configs.Detector, tuple[np.ndarray, np.ndarray]],
    transmission: np.ndarray | None,
) -> dict[configs.Detector, DetectorCube]:
    """Refuse everything each detector holds that is not measured, dropping empty ones.

    Args:
        detectors: Each detector's cube and wavelength table, as read off disk.
        transmission: The atmosphere's transmission over the infrared scan, or None
            when the observation has no infrared half.

    Returns:
        detectors: Every detector that measured, its cube filled and with its mask.

    Raises:
        ValueError: When a window keeps no band of a cube.
    """
    cleaned = {}
    for name, (cube, table) in detectors.items():
        centres = configs.band_centres(table)
        try:
            mask = refused_mask(cube, table, centres, name)
        except NoMeasurement:
            continue
        if name == configs.Detector.INFRARED:
            atmosphere.remove_atmosphere(cube, mask, transmission, centres)
        mask = atmosphere.atmospheric_mask(cube, mask, centres, name)
        mask = remove_outliers.flat_fielded_mask(cube, mask)
        # Despike only the bands in play, so filled ones cannot pull the median about.
        kept = ~mask.bands
        block = np.ascontiguousarray(cube[:, :, kept])
        remove_outliers.remove_spikes(block, mask.pixels)
        cube[:, :, kept] = block
        cleaned[name] = DetectorCube(cube, table, mask)
    return cleaned


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
    np.copyto(cube, remove_outliers.FILL, where=(valid & ~lit)[:, :, None] & bands)
    return kept
