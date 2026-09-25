"""Cutting one observation to a tile: what its box keeps, and where that sits."""

from __future__ import annotations

from building.preprocessing.common import equatorial, geometry, polar
from building.preprocessing.common.models.overlap import Overlap
from building.preprocessing.common.models.relative_position import (
    RelativePosition,
)
from building.preprocessing.common.models.samples import Samples
from common.maths import geodesy
from common.models.tile import Tile


def placed(samples: Samples, frame: Tile) -> RelativePosition:
    """Return where the samples one cut keeps sit, on the grid their tile is read on.

    Args:
        samples: The samples the cut keeps, on the grid they were placed on.
        frame: The tile's local frame, whose centre the offsets stand from.

    Returns:
        position: The offsets from that centre, in polar metres or degrees.
    """
    place = frame.grid
    if place is None:
        return equatorial.placed(samples, frame)
    return polar.placed(samples, frame, place)


def overlap(samples: Samples, frame: Tile) -> Overlap | None:
    """Return what one tile's box keeps of one observation, on the tile's own grid.

    Args:
        samples: The samples of the observation, on the grid it was placed on.
        frame: The tile's local frame, carrying the box the catalogue gives it.

    Returns:
        held: What the box keeps, or None where the observation reaches none of it.
    """
    span = geodesy.longitude_span(frame.west_lon, frame.east_lon)
    # A cut is made where the samples sit; a placement is made where the tile is.
    held = (
        polar.cut(samples, frame, span)
        if samples.grid is not None
        else equatorial.cut(samples, frame, span)
    )
    if held is None:
        return None
    bounds, inside = held
    if samples.separable:
        kept = samples.down[bounds[0]], samples.across[bounds[1]]
    else:
        kept = (
            geometry.taken(samples.down, bounds),
            geometry.taken(samples.across, bounds),
        )
    return Overlap(
        bounds, inside, placed(Samples(*kept, samples.separable, samples.grid), frame)
    )
