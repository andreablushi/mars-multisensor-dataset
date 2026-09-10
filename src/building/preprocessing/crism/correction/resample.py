"""Reading every column onto the survey's own grid, which is where its smile goes."""

from __future__ import annotations

import numpy as np

from building.preprocessing.crism.correction import bands_calibration
from building.preprocessing.crism.models.mask import Mask


def resample_bands(
    cube: np.ndarray, mask: Mask, table: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Read one detector's spectra onto the grid its bands are nominally centred on.

    Args:
        cube: The values as lines by samples by bands, already cleaned, read
            rather than changed.
        mask: What that cleaning refused, whose kept bands are the only ones read
            from and whose fill stands in for a column holding too few of them.
        table: The centre wavelength of every column and band, which every column
            is read off its own row of, so the smile across the detector is taken out.
        grid: The nominal centre of every band of this detector, ascending.

    Returns:
        values: The values as lines by samples by grid bands, each read between
            the two bands of its own column around it and held at the ends.
        measured: One flag per grid band, True where a kept band of the detector
            falls nearer to it than to either of its neighbours.
    """
    kept = ~mask.bands
    read = np.full((*cube.shape[:2], grid.size), mask.fill, dtype="f4")
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
        read[:, at] = held[:, live[low]] * (1.0 - share) + held[:, live[high]] * share

    steps = np.diff(grid)
    borders = np.concatenate(
        ([grid[0] - steps[0] / 2], grid[:-1] + steps / 2, [grid[-1] + steps[-1] / 2])
    )
    measured = np.histogram(bands_calibration.centres(table)[kept], borders)[0] > 0
    return read, measured
