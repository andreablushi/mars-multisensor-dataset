"""Joining the visible and infrared halves, and the geometry beside them."""

from __future__ import annotations

import numpy as np

from building.preprocessing.crism.correction import bands_calibration
from building.preprocessing.crism.models.detector import Detector
from building.preprocessing.crism.models.observation import CrismObservation

# Which detector carries which half of the spectrum.
VISIBLE = "s"
INFRARED = "l"


def merge_detectors(
    identifier: str, detectors: dict[str, Detector]
) -> CrismObservation:
    """Join both detectors of a cleaned observation into one cube.

    Args:
        identifier: The observation the two detectors are halves of.
        detectors: Both halves, already through `read.clean_detectors`, so each
            carries the mask saying what it kept.

    Returns:
        The joined observation, its bands ascending in wavelength.

    Raises:
        ValueError: When the observation has not been cleaned.
    """
    visible, infrared = detectors[VISIBLE], detectors[INFRARED]
    if visible.mask is None or infrared.mask is None:
        raise ValueError(f"{identifier} has not been cleaned.")

    # Only the samples neither detector refused, which is one unbroken run.
    columns = ~(visible.mask.columns | infrared.mask.columns)

    cubes, tables = [], []
    for detector in (visible, infrared):
        bands = ~detector.mask.bands
        cubes.append(detector.cube[:, columns][:, :, bands])
        tables.append(detector.wavelengths[columns][:, bands])

    cube = np.concatenate(cubes, axis=2)
    table = np.concatenate(tables, axis=1)
    # The two overlap around a micron, so ordering is a sort and not a join.
    order = np.argsort(bands_calibration.centres(table))
    return CrismObservation(
        identifier,
        cube[:, :, order],
        table[:, order],
        infrared.geometry[:, columns],
        np.flatnonzero(columns),
    )
