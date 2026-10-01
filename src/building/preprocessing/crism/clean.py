"""Refusing everything a CRISM detector holds that is not a measurement."""

from __future__ import annotations

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism.correction import (
    atmospheric,
    bands_calibration,
    despike,
    destripe,
    ratio,
)
from building.preprocessing.crism.correction.mask import NoMeasurement, refused_mask
from building.preprocessing.crism.models.detector_cube import DetectorCube


def clean_detectors(
    detectors: dict[configs.Detector, tuple[np.ndarray, np.ndarray]],
) -> dict[configs.Detector, DetectorCube]:
    """Refuse everything each detector holds that is not measured, dropping empty ones.

    Args:
        detectors: Each detector's cube and wavelength table, as read off disk.

    Returns:
        detectors: Every detector that measured, its cube filled and with its mask.

    Raises:
        ValueError: When a window keeps no band of a cube.
    """
    cleaned = {}
    for name, (cube, table) in detectors.items():
        centres = bands_calibration.band_centres(table)
        try:
            mask = refused_mask(cube, table, centres, name)
        except NoMeasurement:
            continue
        mask = atmospheric.atmospheric_mask(cube, mask, centres, name)
        destripe.remove_spike_columns(cube, mask, centres, name)
        mask = ratio.ratioed_mask(cube, mask)
        # Despike only the bands in play, so filled ones cannot pull the median about.
        kept = ~mask.bands
        block = np.ascontiguousarray(cube[:, :, kept])
        despike.remove_spikes(block, centres[kept], mask.pixels)
        cube[:, :, kept] = block
        cleaned[name] = DetectorCube(cube, table, mask)
    return cleaned
