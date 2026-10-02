"""Normalising a SHARAD radargram against its clutter simulation, trace by trace."""

from __future__ import annotations

import numpy as np


def normalized_power(power: np.ndarray, clutter: np.ndarray) -> np.ndarray:
    """Return a radargram in dB, each trace's peak moved onto the simulation's peak.

    Args:
        power: The radargram's linear power, delay samples by traces.
        clutter: The simulated clutter power on the same grid, zero without echo.

    Returns:
        normalized: Each trace in dB, less the gap between its peak and the
            simulation's peak in dB.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = 10 * np.log10(power)
        normalized += 10 * np.log10(clutter.max(axis=0)) - normalized.max(axis=0)
    return normalized
