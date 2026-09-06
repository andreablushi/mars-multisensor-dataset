"""Cutting one observation down to the ground its own feature covers."""

from __future__ import annotations

import numpy as np

from building.models.feature import FeatureFrame
from building.preprocessing.common.models.overlap import Box, Overlap
from building.preprocessing.common.models.relative_position import RelativePosition
from building.preprocessing.common.relative_positioning import (
    Positioned,
    relative_position,
)
from utils.geometry import geodesy

# The whole turn, which a longitude offset is measured round.
TURN = 360.0


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
    # How far north of the box's southern edge and east of its western one
    # every sample lies, which is all either branch asks of the box.
    upward = (position.north >= box.south) & (position.north <= box.north)
    # Measured round the turn, so the meridian the box may run over is no edge.
    eastward = (position.east - box.west) % TURN
    if position.separable:
        # The box is a rectangle on a grid whose axes run north and east, so
        # each axis is asked on its own and what they keep is exactly the box.
        lines = np.flatnonzero(upward)
        # An axis counts from its own first longitude, and a box running over
        # the meridian keeps two ends of it that are one strip of ground, so
        # ordering by how far east each lies is what joins those ends back up.
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
    # Bounds that neighbour are sliced rather than gathered, which is what all
    # but an axis rejoined across the meridian keeps, and costs nothing to take.
    runs = tuple(
        slice(int(held[0]), int(held[-1]) + 1)
        for held in bounds
        if held.size and np.all(np.diff(held) == 1)
    )
    if len(runs) == len(bounds):
        return array[runs]
    return array[np.ix_(*bounds)] if len(bounds) > 1 else array[bounds[0]]
