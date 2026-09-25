"""The projection a tile away from the poles is read on: degrees from its centre."""

from __future__ import annotations

from functools import partial

import numpy as np

from building.preprocessing.common import geometry
from building.preprocessing.common.models.position import Position
from common.maths import geodesy
from common.maths.geodesy import TURN
from common.models.tile import Tile


def cut(
    position: Position, frame: Tile, span: float
) -> tuple[tuple[np.ndarray, ...], np.ndarray | None] | None:
    """Return which samples of one observation placed in degrees the box keeps.

    Args:
        position: The samples, placed in the degrees this cut reads them as.
        frame: The tile's local frame, carrying the box the catalogue gives it.
        span: How many degrees of longitude that box covers.

    Returns:
        bounds: The samples to keep of each ground axis, outermost first.
        inside: Which kept samples truly fall in the box, or None for all.
        None: Where the observation reaches none of the box.
    """
    # How far north and east of the box's own edges every sample lies.
    upward = (position.north >= frame.min_lat) & (position.north <= frame.max_lat)
    # Measured round the turn, so the meridian the box may run over is no edge.
    eastward = (position.east - frame.west_lon) % TURN
    if position.separable:
        # The box is a rectangle here, so each axis is asked alone and keeps exactly it.
        lines = np.flatnonzero(upward)
        # A box over the meridian keeps two ends of one strip, joined by ordering east.
        held = np.flatnonzero(eastward <= span)
        ordered = held[np.argsort(eastward[held], kind="stable")]
        if not lines.size or not ordered.size:
            return None
        return (lines, ordered), None
    kept = upward & (eastward <= span)
    if not kept.any():
        return None
    where = np.argwhere(kept)
    bounds = tuple(
        np.arange(int(low), int(high) + 1)
        for low, high in zip(where.min(axis=0), where.max(axis=0), strict=True)
    )
    return bounds, geometry.partial_mask(geometry.kept_part(kept, bounds))


def placed(position: Position, frame: Tile) -> Position:
    """Return where the samples one cut keeps sit, in degrees from the tile centre.

    Args:
        position: The samples the cut keeps, on the grid they were placed on.
        frame: The tile's local frame, whose centre the offsets stand from.

    Returns:
        position: The degrees north and east of that centre.
    """
    if position.grid is None:
        return Position(
            position.north - frame.centre_lat,
            geodesy.normalise_longitude(position.east - frame.centre_lon),
            position.separable,
        )
    north, east = geometry.filled_offsets(
        position, partial(block_offsets, position, frame)
    )
    return Position(north, east, False)


def block_offsets(
    position: Position, frame: Tile, block: tuple
) -> tuple[np.ndarray, np.ndarray]:
    """Return the degrees one block of samples stands from the tile centre.

    Args:
        position: The samples the cut keeps, on the grid they were placed on.
        frame: The tile's local frame, whose centre the offsets stand from.
        block: Which lines of the cut to read.

    Returns:
        north: Their degrees north of the centre.
        east: Their degrees east of it.
    """
    lon, lat = geometry.block_degrees(position, block)
    return lat - frame.centre_lat, geodesy.normalise_longitude(lon - frame.centre_lon)
