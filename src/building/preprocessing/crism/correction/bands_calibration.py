"""Putting one detector's bands in wavelength order and marking what is blank."""

from __future__ import annotations

import numpy as np


def calibrated_cube(
    cube: np.ndarray, table: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Order one cube by wavelength and fill what was never calibrated.

    Args:
        cube: The values as lines by samples by bands, in stored band order.
        table: The centre wavelength of every column and band, NaN if uncalibrated.

    Returns:
        cube: The cube with bands ascending, uncalibrated columns and bands NaN.
        wavelengths: The centre wavelength of every column and band, in that same order.

    Raises:
        ValueError: When the table does not fit the cube or has no calibrated band.
    """
    if cube.shape[1:] != table.shape:
        raise ValueError(
            f"A cube of {cube.shape[1]} columns by {cube.shape[2]} bands cannot "
            f"be read with a table of {table.shape[0]} by {table.shape[1]}."
        )
    # What every band of this detector is centred on, averaged over its columns.
    centre = band_centres(table)
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


def band_centres(table: np.ndarray) -> np.ndarray:
    """Return the centre wavelength of every band, averaged over its columns.

    Args:
        table: The centre wavelength of every column and band.

    Returns:
        centres: One centre per band averaged over its columns, NaN where none.
    """
    # Bands the detector was calibrated for in at least one column.
    named = ~np.isnan(table).all(axis=0)
    # Averaging only those avoids taking the mean of an empty slice.
    out = np.full(table.shape[1], np.nan)
    out[named] = np.nanmean(table[:, named], axis=0)
    return out
