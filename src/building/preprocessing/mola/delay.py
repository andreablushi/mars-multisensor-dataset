"""Placing MOLA heights on the delay rows of a SHARAD radargram."""

from __future__ import annotations

import numpy as np

from building.configs import sharad as sharad_configs
from shared.maths import physics

ROW_UNIT = "RADARGRAM ROW"


def radargram_rows(topography: np.ndarray) -> np.ndarray:
    """Return the radargram row every height's nadir surface echo lands on.

    Args:
        topography: Heights of the ground above the areoid in metres.

    Returns:
        rows: One whole row per height, counted from zero like the radargram and
            wrapped into its window as the radargram wraps high terrain.
    """
    metres_per_row = physics.SPEED_OF_LIGHT_M_S * sharad_configs.DELAY_INTERVAL_S / 2.0
    rows = np.trunc(sharad_configs.AREOID_ROW - topography / metres_per_row)
    return (rows % sharad_configs.DELAY_ROWS).astype(np.int16)


def row_label(label: dict[str, str]) -> dict[str, str]:
    """Return a MOLA label whose unit names the rows it now holds.

    Args:
        label: The merged label of the sheets the rows were read from.

    Returns:
        label: The same label, its unit replaced by the radargram row.
    """
    return {**label, "UNIT": ROW_UNIT}
