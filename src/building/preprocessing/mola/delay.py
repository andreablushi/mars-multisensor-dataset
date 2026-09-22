"""Placing MOLA heights on the delay rows of a SHARAD radargram."""

from __future__ import annotations

import numpy as np

from building.configs import sharad as sharad_configs
from common.maths import physics


def radargram_rows(topography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the radargram row every height's nadir surface echo lands on.

    Args:
        topography: Heights of the ground above the areoid in metres.

    Returns:
        rows: One radargram row per height, held to its window.
        inside: Which of them the window holds, false where held to an edge.
    """
    metres_per_row = physics.SPEED_OF_LIGHT_M_S * sharad_configs.DELAY_INTERVAL_S / 2.0
    rows = np.trunc(sharad_configs.AREOID_ROW - topography / metres_per_row)
    inside = (rows >= 0) & (rows < sharad_configs.DELAY_ROWS)
    held = np.clip(rows, 0, sharad_configs.DELAY_ROWS - 1).astype(np.int16)
    return held, inside
