"""The projection a tile away from the poles is read on: degrees from its centre."""

from __future__ import annotations

import numpy as np

from building.preprocessing.common.crop import marked, sample_sizes, sampled, taken
from building.preprocessing.common.models.relative_position import (
    PolarGrid,
    RelativePosition,
)
from shared.maths import geodesy
from shared.maths.geodesy import TURN
from shared.models.tile import Tile

# How many samples of a cut are crossed at once, since a grid can be huge.
BLOCK = 4_000_000


def cut(
    down: np.ndarray, across: np.ndarray, separable: bool, frame: Tile, span: float
) -> tuple[tuple[np.ndarray, ...], np.ndarray | None] | None:
    """Return which samples of one observation placed in degrees the box keeps.

    Args:
        down: The latitude of every line in degrees, one per sample where the
            two are not separable.
        across: The longitude of every sample, holding the same.
        separable: Whether those two hold one axis each rather than a value for
            every sample.
        frame: The tile's local frame, carrying the box the catalogue gives it.
        span: How many degrees of longitude that box covers.

    Returns:
        bounds: The samples to keep of each ground axis, outermost first.
        inside: Which of them truly falls in the box, and None where every one
            of them does.
    """
    north = down - frame.centre_lat
    east = geodesy.normalise_longitude(across - frame.centre_lon)
    west = geodesy.normalise_longitude(frame.west_lon - frame.centre_lon)
    # How far north and east of the box's own edges every sample lies.
    upward = (north >= frame.min_lat - frame.centre_lat) & (
        north <= frame.max_lat - frame.centre_lat
    )
    # Measured round the turn, so the meridian the box may run over is no edge.
    eastward = (east - west) % TURN
    if separable:
        # The box is a rectangle here, so each axis is asked alone and keeps exactly it.
        lines = np.flatnonzero(upward)
        # A box over the meridian keeps two ends of one strip, joined by ordering east.
        held = np.flatnonzero(eastward <= span)
        samples = held[np.argsort(eastward[held], kind="stable")]
        if not lines.size or not samples.size:
            return None
        return (lines, samples), None
    kept = upward & (eastward <= span)
    if not kept.any():
        return None
    where = np.argwhere(kept)
    bounds = tuple(
        np.arange(int(low), int(high) + 1)
        for low, high in zip(where.min(axis=0), where.max(axis=0), strict=True)
    )
    return bounds, marked(taken(kept, bounds))


def placed(
    down: np.ndarray,
    across: np.ndarray,
    separable: bool,
    frame: Tile,
    grid: PolarGrid | None,
) -> RelativePosition:
    """Return where the samples one cut keeps sit, in degrees from the tile centre.

    Args:
        down: The latitude of every line it keeps in degrees, or its northing in
            the metres of `grid`, one per sample where the two are not separable.
        across: The longitude of every sample it keeps, or its easting, holding
            the same.
        separable: Whether those two hold one axis each rather than a value for
            every sample.
        frame: The tile's local frame, whose centre the offsets stand from.
        grid: The grid the samples sit on, and None where they are degrees
            already, which is what leaves the offsets separable.

    Returns:
        position: The degrees north and east of that centre.
    """
    if grid is None:
        return RelativePosition(
            down - frame.centre_lat,
            geodesy.normalise_longitude(across - frame.centre_lon),
            separable,
        )
    sizes = sample_sizes(down, across, separable)
    north = np.empty(sizes, dtype=float)
    east = np.empty(sizes, dtype=float)
    for block, lon, lat in sampled(down, across, separable, grid, sizes, BLOCK):
        north[block] = lat - frame.centre_lat
        east[block] = geodesy.normalise_longitude(lon - frame.centre_lon)
    return RelativePosition(north, east, False)
