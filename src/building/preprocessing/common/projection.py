"""Which projection a crop is read on: its tile's own, equatorial or polar."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common import equatorial, polar
from building.preprocessing.common.crop import taken
from building.preprocessing.common.models.overlap import Overlap
from building.preprocessing.common.models.relative_position import (
    PolarGrid,
    RelativePosition,
)
from shared.maths import geodesy, physics
from shared.models.tile import Tile

# From this latitude up a tile is read on its pole, below it in degrees.
POLAR_LATITUDE = 70.0


def tile_grid(frame: Tile) -> PolarGrid | None:
    """Return the grid one tile is read on, whatever instrument is read for it.

    Args:
        frame: The tile's local frame, whose box settles which projection it is
            read on.

    Returns:
        grid: The stereographic grid of the pole its whole box lies at, centred
            on the meridian, and None where the tile is read in degrees instead.
    """
    if frame.min_lat >= POLAR_LATITUDE:
        return (0.0, True, physics.EQUATORIAL_RADIUS_M)
    if frame.max_lat <= -POLAR_LATITUDE:
        return (0.0, False, physics.EQUATORIAL_RADIUS_M)
    return None


def placed(
    down: np.ndarray,
    across: np.ndarray,
    separable: bool,
    frame: Tile,
    grid: PolarGrid | None = None,
) -> RelativePosition:
    """Return where the samples one cut keeps sit, on the grid their tile is read on.

    Args:
        down: The latitude of every line it keeps in degrees, or its northing in
            the metres of `grid`, one per sample where the two are not separable.
        across: The longitude of every sample it keeps, or its easting, holding
            the same.
        separable: Whether those two hold one axis each rather than a value for
            every sample.
        frame: The tile's local frame, whose centre the offsets stand from.
        grid: The grid the samples sit on, and None where they are degrees.

    Returns:
        position: The offsets from that centre, in the metres of the tile's pole
            where it has one and in degrees where it has none.
    """
    place = tile_grid(frame)
    if place is None:
        return equatorial.placed(down, across, separable, frame, grid)
    return polar.placed(down, across, separable, frame, grid, place)


def overlap(
    down: np.ndarray,
    across: np.ndarray,
    separable: bool,
    frame: Tile,
    grid: PolarGrid | None = None,
) -> Overlap | None:
    """Return what one tile's box keeps of one observation, on the tile's own grid.

    Args:
        down: The latitude of every line in degrees, or its northing in the
            metres of `grid`, one per sample where the two are not separable.
        across: The longitude of every sample, or its easting, holding the same.
        separable: Whether those two hold one axis each rather than a value for
            every sample.
        frame: The tile's local frame, carrying the box the catalogue gives it.
        grid: The grid the two are measured on, and None for degrees.

    Returns:
        held: What the box keeps, or None where the observation reaches none of it.
    """
    span = geodesy.longitude_span(frame.west_lon, frame.east_lon)
    # A cut is made where the samples sit; a placement is made where the tile is.
    held = (
        polar.cut(down, across, grid, frame, span)
        if grid is not None
        else equatorial.cut(down, across, separable, frame, span)
    )
    if held is None:
        return None
    bounds, inside = held
    if grid is not None or separable:
        kept = down[bounds[0]], across[bounds[1]]
        return Overlap(bounds, inside, placed(*kept, True, frame, grid))
    kept = taken(down, bounds), taken(across, bounds)
    return Overlap(bounds, inside, placed(*kept, False, frame, grid))
