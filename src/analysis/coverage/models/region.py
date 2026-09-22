"""The tile a coverage measurement is made against, projected once."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True, slots=True)
class TileRegion:
    """One tile's bounding box, projected into equal-area metres.

    Attributes:
        centre_lon: The projection centre longitude in degrees.
        centre_lat: The projection centre latitude in degrees.
        shape: The bounding box as a polygon in equal-area metres.
        area_m2: The area of that box in square metres.
        tight: The box in lon/lat degrees, which a footprint with area is cut to.
        wide: The same box widened, which a track is cut to before it is buffered.
        polar: The box in polar stereographic metres, or None where not poleward.
        polar_wide: The same box widened, or None for the same reason.
        north: Whether the tile lies north of the equator.
    """

    centre_lon: float
    centre_lat: float
    shape: BaseGeometry
    area_m2: float
    tight: BaseGeometry
    wide: BaseGeometry
    polar: BaseGeometry | None
    polar_wide: BaseGeometry | None
    north: bool
