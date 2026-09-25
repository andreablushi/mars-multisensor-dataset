"""Cutting one observation to a tile: what its box keeps, and where that sits."""

from __future__ import annotations

from dataclasses import replace

from building.preprocessing.common import equatorial, geometry, polar
from building.preprocessing.common.models.overlap import Overlap
from building.preprocessing.common.models.position import Position
from common.maths import geodesy
from common.models.tile import Tile


def placed(position: Position, frame: Tile) -> Position:
    """Return where the samples one cut keeps sit, on the grid their tile is read on.

    Args:
        position: The samples the cut keeps, on the grid they were placed on.
        frame: The tile's local frame, whose centre the offsets stand from.

    Returns:
        position: The offsets from that centre, in polar metres or degrees.
    """
    if frame.grid is None:
        return equatorial.placed(position, frame)
    return polar.placed(position, frame)


def overlap(position: Position, frame: Tile) -> Overlap | None:
    """Return what one tile's box keeps of one observation, on the tile's own grid.

    Args:
        position: The samples of the observation, on the grid it was placed on.
        frame: The tile's local frame, carrying the box the catalogue gives it.

    Returns:
        held: What the box keeps, or None where the observation reaches none of it.
    """
    span = geodesy.longitude_span(frame.west_lon, frame.east_lon)
    # A cut is made where the samples sit; a placement is made where the tile is.
    held = (
        polar.cut(position, frame, span)
        if position.grid is not None
        else equatorial.cut(position, frame, span)
    )
    if held is None:
        return None
    bounds, inside = held
    if position.separable:
        north, east = position.north[bounds[0]], position.east[bounds[1]]
    else:
        north = geometry.kept_part(position.north, bounds)
        east = geometry.kept_part(position.east, bounds)
    kept = replace(position, north=north, east=east)
    return Overlap(bounds, inside, placed(kept, frame))
