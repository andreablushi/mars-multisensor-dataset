"""Which samples of an observation fall on the ground a feature covers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.relative_position import RelativePosition


@dataclass(frozen=True, slots=True)
class Overlap:
    """What one feature's box keeps of one observation.

    Attributes:
        bounds: The samples to keep of each ground axis, in the position's
            order.
        inside: Which of the samples that survives the cut truly falls in the
            box, or None where every one of them does. A map raster meets a box
            in a rectangle and so is left unset, and only a swath crossing the
            box or a track grazing it has corners to mark.
        position: Where the samples that are kept sit, in degrees from the
            feature centre.
    """

    bounds: tuple[np.ndarray, ...]
    inside: np.ndarray | None
    position: RelativePosition
