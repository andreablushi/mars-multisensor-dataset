"""Reading every column onto the survey's own grid, which is where its smile goes."""

from __future__ import annotations

import numpy as np

from building.preprocessing.crism.correction import bands_calibration
from building.preprocessing.crism.models.mask import Mask


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
    return np.histogram(bands_calibration.centres(table)[~mask.bands], borders)[0] > 0


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
        centre = own[live]
        high = np.clip(np.searchsorted(centre, grid), 1, centre.size - 1)
        low = high - 1
        share = np.clip(
            (grid - centre[low]) / (centre[high] - centre[low]), 0.0, 1.0
        ).astype("f4")
        held = cube[:, at, :]
        out[:, at, bands] = (
            held[:, live[low]] * (1.0 - share) + held[:, live[high]] * share
        )
