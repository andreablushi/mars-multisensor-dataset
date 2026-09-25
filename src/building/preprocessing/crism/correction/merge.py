"""Joining the detectors one observation was delivered as, and its geometry."""

from __future__ import annotations

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism.correction import resample
from building.preprocessing.crism.models.detector_cube import DetectorCube
from building.preprocessing.crism.models.observation import CrismObservation


def merge_detectors(
    detectors: dict[configs.Detector, DetectorCube],
    geometry: np.ndarray,
    label: dict[str, str],
) -> CrismObservation:
    """Join the detectors of a cleaned observation onto the survey's own grid.

    Args:
        detectors: The halves that landed, cleaned, each with its own mask.
        geometry: The backplanes that place every pixel, on the same grid.
        label: What every product the observation was published as says of it.

    Returns:
        observation: The joined observation on the survey's whole band grid.
    """
    # Every half is read out from the first frame, so the shortest ends the strip.
    lines = min(geometry.shape[0], *(held.cube.shape[0] for held in detectors.values()))
    # Only the samples no half refused.
    columns = ~np.logical_or.reduce([held.mask.columns for held in detectors.values()])

    joined = np.full(
        (lines, int(columns.sum()), len(configs.BANDS_NM)), np.nan, dtype="f4"
    )
    measured = np.zeros(len(configs.BANDS_NM), dtype=bool)
    for name, held in detectors.items():
        grid = np.asarray(configs.DETECTOR_BANDS_NM[name])
        table = held.wavelengths[columns]
        live = resample.measured_bands(held.mask, table, grid)
        chosen = np.asarray(configs.DETECTOR_SLOTS[name])[live]
        measured[chosen] = True
        resample.resample_bands(
            held.cube[:lines, columns], held.mask, table, grid[live], joined, chosen
        )

    # A pixel any half could not read is no measurement of the observation.
    valid = ~np.logical_or.reduce(
        [held.mask.pixels[:lines] for held in detectors.values()]
    )[:, columns]
    return CrismObservation(label, joined, geometry[:lines, columns], valid, measured)
