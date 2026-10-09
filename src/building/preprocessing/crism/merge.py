"""Joining the detectors one observation was delivered as onto the survey's grid."""

from __future__ import annotations

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism.correction import centre_wavelengths
from building.preprocessing.crism.models.observation import (
    CrismObservation,
    DetectorCube,
)


def merge_detectors(
    detectors: dict[configs.Detector, DetectorCube],
    geometry: np.ndarray,
    label: dict[str, str],
) -> CrismObservation:
    """Join the detectors of a cleaned observation onto the survey's own grid.

    Args:
        detectors: The halves that landed, cleaned.
        geometry: The backplanes that place every pixel, on the same grid.
        label: What every product the observation was published as says of it.

    Returns:
        observation: The joined observation on the survey's whole band grid.
    """
    # Every half is read out from the first frame, so the shortest ends the strip.
    lines = min(geometry.shape[0], *(held.cube.shape[0] for held in detectors.values()))
    # Only the samples no half refused.
    columns = ~np.logical_or.reduce([held.columns for held in detectors.values()])

    joined = np.full(
        (lines, int(columns.sum()), len(centre_wavelengths.BANDS_NM)),
        np.nan,
        dtype="f4",
    )
    measured = np.zeros(len(centre_wavelengths.BANDS_NM), dtype=bool)
    for name, held in detectors.items():
        slots = centre_wavelengths.DETECTOR_SLOTS[name]
        grid = np.asarray(centre_wavelengths.DETECTOR_BANDS_NM[name])
        measured[slots] = True
        cube = held.cube[:lines, columns]
        for at, centres in enumerate(held.table[columns]):
            high = np.clip(np.searchsorted(centres, grid), 1, centres.size - 1)
            low = high - 1
            share = np.clip(
                (grid - centres[low]) / (centres[high] - centres[low]), 0.0, 1.0
            ).astype("f4")
            spectra = cube[:, at, :]
            joined[:, at, slots] = (
                spectra[:, low] * (1.0 - share) + spectra[:, high] * share
            )

    # A pixel any half could not read is no measurement of the observation.
    valid = ~np.logical_or.reduce([held.pixels[:lines] for held in detectors.values()])[
        :, columns
    ]
    return CrismObservation(label, joined, geometry[:lines, columns], valid, measured)
