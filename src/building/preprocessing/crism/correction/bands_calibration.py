"""Putting one detector's bands in wavelength order and marking what is blank."""

from __future__ import annotations

import numpy as np


def calibrate(cube: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order one cube by wavelength and fill what was never calibrated.

    Args:
        cube: The values as lines by samples by bands, in the band order the
            file stored them in.
        table: The centre wavelength of every column and band, in that same
            order, with NaN where nothing was calibrated.

    Returns:
        The cube with its bands ascending in wavelength and its uncalibrated
        columns and bands NaN, and the centre wavelength of every column and
        band in that same order.

    Raises:
        ValueError: When the table does not describe the cube it is given, or
            when it names no calibrated band to read the band order off.
    """
    if cube.shape[1:] != table.shape:
        raise ValueError(
            f"A cube of {cube.shape[1]} columns by {cube.shape[2]} bands cannot "
            f"be read with a table of {table.shape[0]} by {table.shape[1]}."
        )
    # What every column and band of this detector is centred on.
    centre = centres(table)
    # Read the direction off the file instead of assuming one.
    named = np.flatnonzero(~np.isnan(centre))
    if not named.size:
        raise ValueError("No band of this cube was ever calibrated.")
    if centre[named[0]] > centre[named[-1]]:
        cube, table = cube[:, :, ::-1], table[:, ::-1]

    # A writable copy in the new order, since the reversal above is a view.
    ordered = np.array(cube, dtype="f4")
    # Say what was never calibrated with NaN, leaving the shape alone.
    blank = np.isnan(table)
    ordered[:, blank.all(axis=1), :] = np.nan
    ordered[:, :, blank.all(axis=0)] = np.nan
    return ordered, table


def centres(table: np.ndarray) -> np.ndarray:
    """Return the centre wavelength of every band, averaged over its columns.

    Args:
        table: The centre wavelength of every column and band.

    Returns:
        One centre per band, averaged over the columns that carry one, NaN
        where no column does.
    """
    # Bands the detector was calibrated for in at least one column.
    named = ~np.isnan(table).all(axis=0)
    # Averaging only those avoids taking the mean of an empty slice.
    out = np.full(table.shape[1], np.nan)
    out[named] = np.nanmean(table[:, named], axis=0)
    return out


def window(centre: np.ndarray, width: float) -> int:
    """Return how many bands a window of a given width covers.

    Args:
        centre: The centre wavelength of every kept band, in order.
        width: How far the window should reach, in nm.

    Returns:
        An odd band count whose span fits inside the width, never below three,
        which is what a single band is given since it has no step to measure.
    """
    if centre.size < 2:
        return 3
    step = np.abs(np.diff(centre)).mean()
    size = int(width // step) + 1
    return max(size - 1 + size % 2, 3)
