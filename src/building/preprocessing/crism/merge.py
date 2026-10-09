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
        measured[centre_wavelengths.DETECTOR_SLOTS[name]] = True
        resample_bands(
            held.cube[:lines, columns],
            held.table[columns],
            np.asarray(centre_wavelengths.DETECTOR_BANDS_NM[name]),
            joined,
            centre_wavelengths.DETECTOR_SLOTS[name],
        )

    # A pixel any half could not read is no measurement of the observation.
    valid = ~np.logical_or.reduce([held.pixels[:lines] for held in detectors.values()])[
        :, columns
    ]
    return CrismObservation(label, joined, geometry[:lines, columns], valid, measured)


def resample_bands(
    cube: np.ndarray,
    table: np.ndarray,
    grid: np.ndarray,
    out: np.ndarray,
    bands: list[int],
) -> None:
    """Read one detector's spectra onto the grid its bands are nominally centred on.

    Args:
        cube: The cleaned values as lines by samples by bands, read only.
        table: The centre wavelength of every column and band, per column.
        grid: The nominal centres to read, ascending.
        out: The lines by samples by bands written into.
        bands: Where each grid band lands along the last axis of `out`.
    """
    for at in range(cube.shape[1]):
        centres = table[at]
        high = np.clip(np.searchsorted(centres, grid), 1, centres.size - 1)
        low = high - 1
        share = np.clip(
            (grid - centres[low]) / (centres[high] - centres[low]), 0.0, 1.0
        ).astype("f4")
        held = cube[:, at, :]
        out[:, at, bands] = held[:, low] * (1.0 - share) + held[:, high] * share
