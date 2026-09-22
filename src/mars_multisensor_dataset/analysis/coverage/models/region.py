"""The tile a coverage measurement is made against, projected once."""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True, slots=True)
class TileRegion:
    """One tile's bounding box, projected into equal-area metres.

    Attributes:
        centre_lon: The projection centre longitude in degrees.
        centre_lat: The projection centre latitude in degrees.
        laea: The bounding box as a polygon in equal-area metres.
        clip: The box in lon/lat degrees, which a footprint with area is cut to.
        clip_wide: The same box widened, which a track is cut to before it is buffered.
        polar_clip: The box in ODE's polar stereographic metres, or None where not
            poleward.
        polar_clip_wide: The same box widened, or None for the same reason.
        north: Whether the tile lies north of the equator.
    """

    centre_lon: float
    centre_lat: float
    laea: BaseGeometry
    clip: BaseGeometry
    clip_wide: BaseGeometry
    polar_clip: BaseGeometry | None
    polar_clip_wide: BaseGeometry | None
    north: bool

    @property
    def span_m(self) -> float:
        """Return the side of a square as large as the box's bounds.

        Returns:
            metres: The side in equal-area metres.
        """
        west, south, east, north = self.laea.bounds
        return math.sqrt((east - west) * (north - south))
