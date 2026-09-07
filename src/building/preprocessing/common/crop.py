"""Cutting one observation down to the ground its own feature covers."""

from __future__ import annotations

import numpy as np

from building.models.feature import FeatureFrame
from building.preprocessing.common.models.overlap import Box, Overlap
from building.preprocessing.common.models.relative_position import RelativePosition
from building.preprocessing.common.relative_positioning import (
    Positioned,
    Projected,
    relative_position,
)
from utils.geometry import geodesy

# The whole turn, which a longitude offset is measured round.
TURN = 360.0

# The longest segment the box is walked in, a chord leaving its arc by under a pixel.
STEP = 0.1

# How many pixels of a polar cut become degrees at once, since a grid can be huge.
BLOCK = 4_000_000


def overlap(observation: Positioned, frame: FeatureFrame) -> Overlap | None:
    """Return what one feature's box keeps of one observation.

    Args:
        observation: The observation as it was read off disk, saying where its
            own samples were measured.
        frame: The feature's local frame, carrying the box the catalogue gives
            it, which is read as the same degrees from that centre.

    Returns:
        What the box keeps, or None where the observation reaches none of it.
    """
    position = relative_position(observation, frame)
    box = Box(
        south=frame.min_lat - frame.centre_lat,
        north=frame.max_lat - frame.centre_lat,
        west=geodesy.normalise_longitude(frame.west_lon - frame.centre_lon),
        span=geodesy.longitude_span(frame.west_lon, frame.east_lon),
    )
    # How far north and east of the box's own edges every sample lies.
    upward = (position.north >= box.south) & (position.north <= box.north)
    # Measured round the turn, so the meridian the box may run over is no edge.
    eastward = (position.east - box.west) % TURN
    if position.separable:
        # The box is a rectangle here, so each axis is asked alone and keeps exactly it.
        lines = np.flatnonzero(upward)
        # A box over the meridian keeps two ends of one strip, joined by ordering east.
        held = np.flatnonzero(eastward <= box.span)
        samples = held[np.argsort(eastward[held], kind="stable")]
        if not lines.size or not samples.size:
            return None
        return Overlap(
            (lines, samples),
            None,
            RelativePosition(position.north[lines], position.east[samples], True),
        )
    inside = upward & (eastward <= box.span)
    if not inside.any():
        return None
    where = np.argwhere(inside)
    bounds = tuple(
        np.arange(int(low), int(high) + 1)
        for low, high in zip(where.min(axis=0), where.max(axis=0), strict=True)
    )
    return Overlap(
        bounds,
        marked(taken(inside, bounds)),
        RelativePosition(
            taken(position.north, bounds), taken(position.east, bounds), False
        ),
    )


def marked(held: np.ndarray) -> np.ndarray | None:
    """Return one mask, or nothing at all where it marks every sample.

    Args:
        held: The mask over the samples a crop keeps.

    Returns:
        The mask, or None where every sample of it is true and so it says
        nothing the shape does not already.
    """
    return None if held.all() else held


def taken(array: np.ndarray, bounds: tuple[np.ndarray, ...]) -> np.ndarray:
    """Return the part of one array a cut's bounds keep of its leading axes.

    Args:
        array: The array to cut, whose leading axes are the ground's.
        bounds: The samples to keep of each of those axes.

    Returns:
        The part that is left, every axis past the ground's kept whole.
    """
    # Neighbouring bounds are sliced rather than gathered, which costs nothing to take.
    runs = tuple(
        slice(int(held[0]), int(held[-1]) + 1)
        for held in bounds
        if held.size and np.all(np.diff(held) == 1)
    )
    if len(runs) == len(bounds):
        return array[runs]
    return array[np.ix_(*bounds)] if len(bounds) > 1 else array[bounds[0]]


def polar_overlap(observation: Projected, frame: FeatureFrame) -> Overlap | None:
    """Return what one feature's box keeps of one grid projected onto a pole.

    Args:
        observation: The grid as its label projects it, in the projection's own
            metres.
        frame: The feature's local frame, carrying the box the catalogue gives
            it.

    Returns:
        What the box keeps, or None where the grid reaches none of it.
    """
    grid = observation.polar
    ring = geodesy.stereographic_forward(
        *geodesy.bbox_ring(
            frame.min_lat, frame.max_lat, frame.west_lon, frame.east_lon, STEP
        ),
        *grid,
    )
    # The box projects to a sector, and the ring its edge traces bounds it.
    lines = np.flatnonzero(
        (observation.down >= ring[1].min()) & (observation.down <= ring[1].max())
    )
    samples = np.flatnonzero(
        (observation.across >= ring[0].min()) & (observation.across <= ring[0].max())
    )
    if not lines.size or not samples.size:
        return None
    # Only the sector's rectangle is crossed back, a block of its lines at a time.
    span = geodesy.longitude_span(frame.west_lon, frame.east_lon)
    across = observation.across[samples][None, :]
    inside = np.empty((lines.size, samples.size), dtype=bool)
    reach = max(1, BLOCK // samples.size)
    for start in range(0, lines.size, reach):
        block = slice(start, start + reach)
        lon, lat = geodesy.stereographic_inverse(
            across, observation.down[lines[block]][:, None], *grid
        )
        inside[block] = (
            (lat >= frame.min_lat)
            & (lat <= frame.max_lat)
            & ((lon - frame.west_lon) % TURN <= span)
        )
    if not inside.any():
        return None
    centre_x, centre_y = geodesy.stereographic_forward(
        frame.centre_lon, frame.centre_lat, *grid
    )
    return Overlap(
        (lines, samples),
        marked(inside),
        RelativePosition(
            observation.down[lines] - float(centre_y),
            observation.across[samples] - float(centre_x),
            True,
            grid,
        ),
    )
