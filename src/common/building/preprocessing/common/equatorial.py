"""The projection a tile away from the poles is read on: degrees from its centre."""

from __future__ import annotations

import numpy as np

from common.building.preprocessing.common import geometry
from common.building.preprocessing.common.models.cut import Cut
from common.building.preprocessing.common.models.relative_position import (
    RelativePosition,
)
from common.building.preprocessing.common.models.samples import Samples
from common.maths import geodesy
from common.maths.geodesy import TURN
from common.models.tile import Tile


def cut(samples: Samples, frame: Tile, span: float) -> Cut | None:
    """Return which samples of one observation placed in degrees the box keeps.

    Args:
        samples: The samples, placed in the degrees this cut reads them as.
        frame: The tile's local frame, carrying the box the catalogue gives it.
        span: How many degrees of longitude that box covers.

    Returns:
        held: What the box keeps, or None where the observation reaches none of it.
    """
    # How far north and east of the box's own edges every sample lies.
    upward = (samples.down >= frame.min_lat) & (samples.down <= frame.max_lat)
    # Measured round the turn, so the meridian the box may run over is no edge.
    eastward = (samples.across - frame.west_lon) % TURN
    if samples.separable:
        # The box is a rectangle here, so each axis is asked alone and keeps exactly it.
        lines = np.flatnonzero(upward)
        # A box over the meridian keeps two ends of one strip, joined by ordering east.
        held = np.flatnonzero(eastward <= span)
        ordered = held[np.argsort(eastward[held], kind="stable")]
        if not lines.size or not ordered.size:
            return None
        return Cut((lines, ordered), None, True)
    kept = upward & (eastward <= span)
    if not kept.any():
        return None
    where = np.argwhere(kept)
    bounds = tuple(
        np.arange(int(low), int(high) + 1)
        for low, high in zip(where.min(axis=0), where.max(axis=0), strict=True)
    )
    return Cut(bounds, geometry.marked(geometry.taken(kept, bounds)), False)


def placed(samples: Samples, frame: Tile) -> RelativePosition:
    """Return where the samples one cut keeps sit, in degrees from the tile centre.

    Args:
        samples: The samples the cut keeps, on the grid they were placed on.
        frame: The tile's local frame, whose centre the offsets stand from.

    Returns:
        position: The degrees north and east of that centre, separable only where
            the samples are degrees already.
    """
    if samples.grid is None:
        return RelativePosition(
            samples.down - frame.centre_lat,
            geodesy.normalise_longitude(samples.across - frame.centre_lon),
            samples.separable,
        )

    def offsets(block: slice) -> tuple[np.ndarray, np.ndarray]:
        """Return the degrees one block of samples stands from that centre.

        Args:
            block: Which lines of the cut to read.

        Returns:
            north: Their degrees north of the centre.
            east: Their degrees east of it.
        """
        lon, lat = geometry.degrees(samples, block)
        return lat - frame.centre_lat, geodesy.normalise_longitude(
            lon - frame.centre_lon
        )

    north, east = geometry.filled(samples, offsets)
    return RelativePosition(north, east, False)
