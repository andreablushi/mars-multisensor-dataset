"""Saying which cells of a cube are not measurements, and filling them."""

from __future__ import annotations

import numpy as np

from building.configs.crism import Detector
from building.preprocessing.crism.correction import bands_calibration
from building.preprocessing.crism.models.mask import Mask

# The nm window each detector is trusted over, outside which the reading is noise.
WINDOWS = {Detector.INFRARED: (1020.0, 2650.0), Detector.VISIBLE: (400.0, 1060.0)}

# The range a brightness can take, its floor below zero so noise there survives.
BRIGHTNESS = (-0.05, 1.0)


class NoMeasurement(ValueError):
    """Raised when every cell of one detector's cube is refused."""


def bad_pixels(cube: np.ndarray, table: np.ndarray, detector: Detector) -> Mask:
    """Fill everything one cube holds that is not a measurement.

    Args:
        cube: The values as lines by samples by bands, filled in place.
        table: The centre wavelength of every column and band, in that order.
        detector: Which detector, `l` for infrared or `s` for visible.

    Returns:
        mask: The mask saying where the cube was filled rather than measured.

    Raises:
        KeyError: When no window is configured for that detector.
        ValueError: When the window keeps no band of the cube.
        NoMeasurement: When no cell of the cube is a measurement.
    """
    centre = bands_calibration.centres(table)
    low, high = WINDOWS[detector]

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
    floor, ceiling = BRIGHTNESS
    scattered = cube < floor
    scattered |= cube > ceiling
    scattered |= ~np.isfinite(cube)
    scattered &= ~dead

    # A pixel is unusable when its column is dead or any of its bands is.
    pixels = scattered.any(axis=2)
    pixels[:, columns] = True

    # One stand-in for every refused cell, read off what the cube still measures.
    refused = scattered | dead
    if refused.all():
        raise NoMeasurement(f"No cell of this {detector} cube is a measurement.")
    fill = float(np.mean(cube, where=~refused))
    np.copyto(cube, fill, where=refused)
    return Mask(columns, bands, pixels, fill)
