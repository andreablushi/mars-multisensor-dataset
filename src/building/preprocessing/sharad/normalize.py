"""Normalising a SHARAD radargram against its clutter simulation, trace by trace."""

from __future__ import annotations

import numpy as np


def normalized_power(power: np.ndarray, clutter: np.ndarray) -> np.ndarray:
    """Return a radargram in dB under its surface echo, its simulated clutter removed.

    Args:
        power: The radargram's linear power, delay samples by traces.
        clutter: The simulated clutter power on the same grid, zero without echo.

    Returns:
        normalized: Each trace in dB below its own peak, less the simulation's trace in
            dB below its peak where it simulates an echo, the two peaks on one row.
    """
    rows = len(power)
    shift = (clutter.argmax(axis=0) - power.argmax(axis=0)).astype(np.int32)
    # Each simulated trace is moved so its peak lands on the radargram's
    source = np.arange(rows, dtype=np.int32)[:, None] + shift
    outside = (source < 0) | (source >= rows)
    aligned = np.take_along_axis(clutter, source.clip(0, rows - 1), axis=0)
    aligned[outside] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = 10 * np.log10(power)
        normalized -= normalized.max(axis=0)
        np.log10(aligned, out=aligned)
        aligned *= 10
        aligned -= aligned.max(axis=0)
    np.subtract(normalized, aligned, out=normalized, where=np.isfinite(aligned))
    return normalized
