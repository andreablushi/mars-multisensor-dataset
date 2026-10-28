"""The delay rows of a stored SHARAD crop below its atmosphere, for a reader to cut."""

from __future__ import annotations

import numpy as np

GUARD_ROWS = 30


def rows_below_atmosphere(power: np.ndarray, measured: np.ndarray) -> slice:
    """Return the delay rows from a guard above a crop's highest surface down.

    Args:
        power: The crop as stored, dB over its surface, delay samples by traces.
        measured: Which of its traces were measured.

    Returns:
        rows: From `GUARD_ROWS` above the strongest echo nearest the top to the last
            row, counted as the archive counts them.
    """
    top = int(power[:, measured].argmax(axis=0).min()) - GUARD_ROWS
    return slice(max(top, 0), len(power))
