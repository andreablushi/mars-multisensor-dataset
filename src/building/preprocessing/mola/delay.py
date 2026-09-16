"""Placing MOLA heights on the delay rows of a SHARAD radargram."""

from __future__ import annotations

import numpy as np

from building.configs import sharad as sharad_configs
from shared.maths import physics


def radargram_rows(topography: np.ndarray) -> np.ndarray:
    """Return the radargram row every height's nadir surface echo lands on.

    Args:
        topography: Heights of the ground above the areoid in metres.

    Returns:
        rows: One fractional row per height, counted from zero like the radargram.
    """
    metres_per_row = physics.SPEED_OF_LIGHT_M_S * sharad_configs.DELAY_INTERVAL_S / 2.0
    return sharad_configs.AREOID_ROW - topography / metres_per_row
