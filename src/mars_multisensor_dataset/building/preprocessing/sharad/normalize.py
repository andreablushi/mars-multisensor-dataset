"""SHARAD power in dB, and the surface a crop's clutter simulation is aligned on."""

from __future__ import annotations

import numpy as np

from mars_multisensor_dataset.building.configs import sharad as configs


def radargram_db(power: np.ndarray) -> np.ndarray:
    """Return a radargram in dB, -inf wherever its power is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(power)


def clutter_db(clutter: np.ndarray) -> np.ndarray:
    """Return a clutter simulation in dB, every bin without echo at the floor.

    Args:
        clutter: The simulated clutter power, zero without echo.

    Returns:
        clutter: The same grid in dB, never below `configs.CLUTTER_FLOOR_DB`.
    """
    floor = np.float32(10 ** (configs.CLUTTER_FLOOR_DB / 10))
    return 10 * np.log10(np.maximum(clutter, floor))


def mean_surface_peak(db: np.ndarray, measured: np.ndarray) -> float:
    """Return the average peak of a crop's measured traces that echo, in dB.

    Args:
        db: One crop in dB, delay samples by traces.
        measured: Which of its traces were measured.

    Returns:
        peak: The mean of their strongest values.
    """
    peaks = db[:, measured].max(axis=0)
    return float(np.mean(peaks, where=peaks > configs.CLUTTER_FLOOR_DB))


def surface_aligned(db: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """Return a crop in dB less the average peak of its measured traces.

    Args:
        db: One crop in dB, delay samples by traces.
        measured: Which of its traces were measured.

    Returns:
        aligned: The same grid, 0 dB at the average peak of the measured traces
            that echo.
    """
    return db - mean_surface_peak(db, measured)
