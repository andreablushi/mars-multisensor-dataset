"""Refusing everything a CRISM detector holds that is not a measurement."""

from __future__ import annotations

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism.correction import (
    atmospheric,
    bands_calibration,
    despike,
    destripe,
    masking,
    ratio,
)
from building.preprocessing.crism.models.detector_cube import DetectorCube


def clean_detector(
    name: configs.Detector, cube: np.ndarray, table: np.ndarray
) -> DetectorCube:
    """Refuse everything one detector holds that is not measured.

    Args:
        name: Which detector it is.
        cube: The values as lines by samples by bands, which every step changes.
        table: The centre wavelength of every column and band.

    Returns:
        detector: The cleaned cube, carrying the mask every step left.

    Raises:
        ValueError: When a window keeps no band of the cube.
        NoMeasurement: When no cell of the cube is a measurement.
    """
    centre = bands_calibration.band_centres(table)
    mask = masking.refused_mask(cube, table, centre, name)
    mask = atmospheric.remove_atmospheric_bands(cube, mask, centre, name)
    destripe.remove_spike_columns(cube, mask, centre, name)
    mask = ratio.ratio_by_column_median(cube, mask)
    # Despike only the bands in play, so filled ones cannot pull the median about.
    kept = ~mask.bands
    block = np.ascontiguousarray(cube[:, :, kept])
    despike.remove_spikes(block, centre[kept], mask.pixels)
    cube[:, :, kept] = block
    return DetectorCube(cube, table, mask)


def clean_detectors(
    identifier: str,
    detectors: dict[configs.Detector, tuple[np.ndarray, np.ndarray]],
) -> dict[configs.Detector, DetectorCube]:
    """Clean every detector of one observation, dropping any that measured nothing.

    Args:
        identifier: The observation the detectors are halves of.
        detectors: Each detector's cube and wavelengths, as read off disk.

    Returns:
        detectors: Every detector that measured, filled and with its mask.

    Raises:
        ValueError: When a window keeps no band of a cube, or no detector measured.
    """
    cleaned = {}
    for name, (cube, table) in detectors.items():
        try:
            cleaned[name] = clean_detector(name, cube, table)
        except masking.NoMeasurement:
            continue
    if not cleaned:
        raise ValueError(f"No detector of {identifier} holds a measurement.")
    return cleaned
