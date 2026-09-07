"""Saying which cells of a cube are not measurements, and filling them."""

from __future__ import annotations

import numpy as np

from building.preprocessing.crism import configs
from building.preprocessing.crism.correction import bands_calibration
from building.preprocessing.crism.models.mask import Mask


def bad_pixels(cube: np.ndarray, table: np.ndarray, detector: str) -> Mask:
    """Fill everything one cube holds that is not a measurement.

    Args:
        cube: The values as lines by samples by bands, ordered by wavelength,
            filled in place.
        table: The centre wavelength of every column and band, in that order.
        detector: Which detector, `l` for infrared or `s` for visible, which
            picks the window.

    Returns:
        The mask saying where the cube was filled rather than measured.

    Raises:
        KeyError: When no window is configured for that detector.
        ValueError: When nothing at all survives the mask, or when no cell of
            it is a measurement to fill the rest from.
    """
    centre = bands_calibration.centres(table)
    low, high = configs.WINDOWS[detector]

    # What the wavelength file refused to name, which is already NaN.
    columns = np.isnan(table).all(axis=1)
    blank = np.isnan(centre)
    # The sensor edges, where the window says the reading is not trusted.
    edges = ~blank & ((centre < low) | (centre > high))
    bands = blank | edges

    # What no value test may look at, held per column and band so it broadcasts.
    dead = np.zeros(cube.shape[1:], dtype=bool)
    dead[columns, :] = True
    dead[:, bands] = True
    if dead.all():
        raise ValueError(f"The {detector} window keeps no band of this cube.")

    # A brightness outside what light can do is not a reading.
    floor, ceiling = configs.BRIGHTNESS
    scattered = cube < floor
    scattered |= cube > ceiling
    scattered |= ~np.isfinite(cube)
    scattered &= ~dead

    # A pixel is unusable when its column is dead or any of its bands is.
    pixels = scattered.any(axis=2)
    pixels[:, columns] = True

    # One stand-in for every refused cell, read off what the cube still measures.
    refused = scattered | dead
    fill = float(np.mean(cube, where=~refused))
    if not np.isfinite(fill):
        raise ValueError(f"No cell of this {detector} cube is a measurement.")
    np.copyto(cube, fill, where=refused)
    return Mask(columns, bands, edges, int(scattered.sum()), pixels, fill)
