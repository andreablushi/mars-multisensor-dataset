"""Joining the detectors one observation was delivered as, and its geometry."""

from __future__ import annotations

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism.correction import resample
from building.preprocessing.crism.models.detector_cube import DetectorCube
from building.preprocessing.crism.models.observation import CrismObservation


def merge_detectors(
    identifier: str,
    detectors: dict[configs.Detector, DetectorCube],
    geometry: np.ndarray,
    label: dict[str, str],
) -> CrismObservation:
    """Join the detectors of a cleaned observation onto the survey's own grid.

    Args:
        identifier: The observation the detectors are halves of.
        detectors: The halves that landed, cleaned, each with its own mask.
        geometry: The backplanes that place every pixel, on the same grid.
        label: What every product the observation was published as says of it.

    Returns:
        observation: The joined observation on the survey's whole band grid.
    """
    halves = tuple(name for name in configs.Detector if name in detectors)

    # Every half is read out from the first frame, so the shortest ends the strip.
    lines = min(geometry.shape[0], *(detectors[name].cube.shape[0] for name in halves))
    # Only the samples no half refused.
    columns = ~np.logical_or.reduce([detectors[name].mask.columns for name in halves])

    joined = np.full(
        (lines, int(columns.sum()), len(configs.BANDS_NM)), np.nan, dtype="f4"
    )
    measured = np.zeros(len(configs.BANDS_NM), dtype=bool)
    for name in halves:
        grid = np.asarray(configs.DETECTOR_BANDS_NM[name])
        slots = np.asarray(configs.DETECTOR_SLOTS[name])
        held = detectors[name]
        live = resample.measured_bands(held.mask, held.wavelengths[columns], grid)
        chosen = slots[live]
        measured[chosen] = True
        resample.resample_bands(
            held.cube[:lines, columns],
            held.mask,
            held.wavelengths[columns],
            grid[live],
            joined,
            chosen,
        )

    # A pixel any half could not read is no measurement of the observation.
    valid = ~np.logical_or.reduce(
        [detectors[name].mask.pixels[:lines] for name in halves]
    )[:, columns]
    return CrismObservation(
        identifier,
        label,
        joined,
        geometry[:lines, columns],
        valid,
        measured,
    )
