"""Aligning a SHARAD radargram and its clutter simulation on their average surface."""

from __future__ import annotations

import numpy as np

from building.configs import sharad as configs


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


def surface_aligned(db: np.ndarray) -> np.ndarray:
    """Return a track in dB less the average of its traces' peaks.

    Args:
        db: One track in dB, delay samples by traces.

    Returns:
        aligned: The same grid, 0 dB at the average peak of the traces that echo.
    """
    peaks = db.max(axis=0)
    return db - np.mean(peaks, where=peaks > configs.CLUTTER_FLOOR_DB)
