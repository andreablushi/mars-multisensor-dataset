"""The ground of one cut: how its samples are kept, crossed and turned to degrees."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from building.preprocessing.common.models.position import Position
from common.maths import geodesy

# How many samples of a cut are crossed at once, since a grid can run to gigabytes.
BLOCK = 4_000_000


def partial_mask(held: np.ndarray) -> np.ndarray | None:
    """Return one mask, or None where it marks every sample."""
    return None if held.all() else held


def kept_part(array: np.ndarray, bounds: tuple[np.ndarray, ...]) -> np.ndarray:
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


def line_blocks(
    sizes: tuple[int, ...], budget: int = BLOCK
) -> Iterator[tuple[slice, ...]]:
    """Yield the lines of one cut in blocks of about as many samples as fit.

    Args:
        sizes: How many samples each ground axis holds.
        budget: How many samples to read at once.

    Yields:
        taken: Which lines of the cut to read, every other ground axis whole.
    """
    rest = (slice(None),) * (len(sizes) - 1)
    reach = max(1, budget // max(1, int(np.prod(sizes[1:], dtype=int))))
    for start in range(0, sizes[0], reach):
        yield (slice(start, start + reach), *rest)


def block_degrees(position: Position, block: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Return the longitude and latitude one block of samples sits at.

    Args:
        position: The samples, on the grid they were placed on.
        block: Which lines of them to read.

    Returns:
        longitude: Their longitudes in degrees.
        latitude: Their latitudes in degrees.
    """
    north, east = position.crossed_part(block)
    if position.grid is None:
        return east, north
    return geodesy.stereographic_inverse(east, north, *position.grid)


def filled_offsets(
    position: Position,
    offsets: Callable[[tuple], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two offsets of every sample, one block of lines at a time.

    Args:
        position: The samples to cross, whose axes settle how much is read at once.
        offsets: What hands back the northings and the eastings of one block.

    Returns:
        north: The northing of every sample, over every ground axis it holds.
        east: The easting of every one of them, holding the same.
    """
    sizes = position.sizes
    north = np.empty(sizes, dtype=float)
    east = np.empty(sizes, dtype=float)
    for block in line_blocks(sizes):
        north[block], east[block] = offsets(block)
    return north, east
