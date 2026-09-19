"""The ground of one cut: what a box keeps, how it is crossed, and where it sits."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from common.building.preprocessing.common import equatorial, polar
from common.building.preprocessing.common.models.overlap import Overlap
from common.building.preprocessing.common.models.relative_position import (
    RelativePosition,
)
from common.building.preprocessing.common.models.samples import Samples
from common.maths import geodesy
from common.models.tile import Tile

# How many samples of a cut are crossed at once, since a grid can run to gigabytes.
BLOCK = 4_000_000


def marked(held: np.ndarray) -> np.ndarray | None:
    """Return one mask, or nothing at all where it marks every sample.

    Args:
        held: The mask over the samples a crop keeps.

    Returns:
        mask: The mask, or None where every sample is true and it says nothing the shape
            does not.
    """
    return None if held.all() else held


def taken(array: np.ndarray, bounds: tuple[np.ndarray, ...]) -> np.ndarray:
    """Return the part of one array a cut's bounds keep of its leading axes.

    Args:
        array: The array to cut, whose leading axes are the ground's.
        bounds: The samples to keep of each of those axes.

    Returns:
        held: The part that is left, every axis past the ground's kept whole.
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


def blocked(sizes: tuple[int, ...], budget: int = BLOCK) -> Iterator[slice]:
    """Walk the lines of one cut in blocks of about as many samples as fit.

    Args:
        sizes: How many samples each ground axis holds.
        budget: How many samples to read at once, which a heavier conversion
            than a projection asks a smaller one of.

    Yields:
        block: Which lines of the cut to read.
    """
    reach = max(1, budget // max(1, int(np.prod(sizes[1:], dtype=int))))
    for start in range(0, sizes[0], reach):
        yield slice(start, start + reach)


def axes(samples: Samples, block: slice) -> tuple[np.ndarray, np.ndarray]:
    """Return what places one block of samples, as its own grid measured it.

    Args:
        samples: The samples, on the grid they were placed on.
        block: Which lines of them to read.

    Returns:
        down: Their northings or latitudes, a column where the two are separable.
        across: Their eastings or longitudes, a row where they are.
    """
    if samples.separable:
        return samples.down[block][:, None], samples.across[None, :]
    return samples.down[block], samples.across[block]


def degrees(samples: Samples, block: slice) -> tuple[np.ndarray, np.ndarray]:
    """Return the longitude and latitude one block of samples sits at.

    Args:
        samples: The samples, on the grid they were placed on.
        block: Which lines of them to read.

    Returns:
        longitude: Their longitudes in degrees.
        latitude: Their latitudes in degrees.
    """
    down, across = axes(samples, block)
    if samples.grid is None:
        return across, down
    return geodesy.stereographic_inverse(across, down, *samples.grid)


def filled(
    samples: Samples,
    offsets: Callable[[slice], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two offsets of every sample, one block of lines at a time.

    Args:
        samples: The samples to cross, whose axes settle how much is read at once.
        offsets: What hands back the northings and the eastings of one block.

    Returns:
        north: The northing of every sample, over every ground axis it holds.
        east: The easting of every one of them, holding the same.
    """
    sizes = samples.sizes
    north = np.empty(sizes, dtype=float)
    east = np.empty(sizes, dtype=float)
    for block in blocked(sizes):
        north[block], east[block] = offsets(block)
    return north, east


def placed(samples: Samples, frame: Tile) -> RelativePosition:
    """Return where the samples one cut keeps sit, on the grid their tile is read on.

    Args:
        samples: The samples the cut keeps, on the grid they were placed on.
        frame: The tile's local frame, whose centre the offsets stand from.

    Returns:
        position: The offsets from that centre, in the metres of the tile's pole
            where it has one and in degrees where it has none.
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
    if held.separable:
        kept = samples.down[held.bounds[0]], samples.across[held.bounds[1]]
    else:
        kept = taken(samples.down, held.bounds), taken(samples.across, held.bounds)
    return Overlap(
        held.bounds,
        held.inside,
        placed(Samples(*kept, held.separable, samples.grid), frame),
    )
