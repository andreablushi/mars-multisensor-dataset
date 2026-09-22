"""Which samples of an observation fall on the ground a tile covers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.relative_position import (
    RelativePosition,
)


@dataclass(frozen=True, slots=True)
class Overlap:
    """What one tile's box keeps of one observation.

    Attributes:
        bounds: The samples to keep of each ground axis, in the position's order.
        inside: Which kept samples truly fall in the box, or None for all.
        position: Where the kept samples sit, in the grid's degrees or metres.
    """

    bounds: tuple[np.ndarray, ...]
    inside: np.ndarray | None
    position: RelativePosition
