"""Joining the detectors one observation was delivered as onto the survey's grid."""

from __future__ import annotations

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism.models.observation import (
    CrismObservation,
    DetectorCube,
    Mask,
)


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
        table = held.table[columns]
        live = measured_bands(held.mask, table, grid)
        chosen = np.asarray(configs.DETECTOR_SLOTS[name])[live]
        measured[chosen] = True
        resample_bands(
            held.cube[:lines, columns], held.mask, table, grid[live], joined, chosen
        )

    # A pixel any half could not read is no measurement of the observation.
    valid = ~np.logical_or.reduce(
        [held.mask.pixels[:lines] for held in detectors.values()]
    )[:, columns]
    return CrismObservation(label, joined, geometry[:lines, columns], valid, measured)


def measured_bands(mask: Mask, table: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return which bands of the grid one detector measured.

    Args:
        mask: What the cleaning refused, whose kept bands are the only ones counted.
        table: The centre wavelength of every column and band.
        grid: The nominal centre of every band of this detector, ascending.

    Returns:
        measured: One flag per grid band, True where a kept band is nearest.
    """
    steps = np.diff(grid)
    borders = np.concatenate(
        ([grid[0] - steps[0] / 2], grid[:-1] + steps / 2, [grid[-1] + steps[-1] / 2])
    )
    kept = configs.band_centres(table)[~mask.bands]
    return np.histogram(kept, borders)[0] > 0


def resample_bands(
    cube: np.ndarray,
    mask: Mask,
    table: np.ndarray,
    grid: np.ndarray,
    out: np.ndarray,
    bands: np.ndarray,
) -> None:
    """Read one detector's spectra onto the grid its bands are nominally centred on.

    Args:
        cube: The cleaned values as lines by samples by bands, read only.
        mask: What that cleaning refused, whose kept bands alone are read.
        table: The centre wavelength of every column and band, per column.
        grid: The nominal centres to read, ascending.
        out: The lines by samples by bands written into.
        bands: Where each grid band lands along the last axis of `out`.
    """
    kept = ~mask.bands
    out[:, :, bands] = mask.fill
    for at in range(cube.shape[1]):
        own = table[at]
        live = np.flatnonzero(kept & ~np.isnan(own))
        if live.size < 2:
            continue
        centres = own[live]
        high = np.clip(np.searchsorted(centres, grid), 1, centres.size - 1)
        low = high - 1
        share = np.clip(
            (grid - centres[low]) / (centres[high] - centres[low]), 0.0, 1.0
        ).astype("f4")
        held = cube[:, at, :]
        out[:, at, bands] = (
            held[:, live[low]] * (1.0 - share) + held[:, live[high]] * share
        )
