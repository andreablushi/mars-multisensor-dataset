"""Which samples of an observation fall on the ground a tile covers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from mars_multisensor_dataset.building.preprocessing.common.models.position import (
    Position,
)


class KeptSamples(NamedTuple):
    """Which samples of one observation a tile's box keeps, before they are placed.

    Attributes:
        bounds: The samples to keep of each ground axis, outermost first.
        inside: Which kept samples truly fall in the box, or None for all.
    """

    bounds: tuple[np.ndarray, ...]
    inside: np.ndarray | None


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
    position: Position
