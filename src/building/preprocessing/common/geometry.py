"""The ground of one cut: how its samples are taken, crossed and turned to degrees."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from building.preprocessing.common.models.samples import Samples
from common.maths import geodesy

# How many samples of a cut are crossed at once, since a grid can run to gigabytes.
BLOCK = 4_000_000


def marked(held: np.ndarray) -> np.ndarray | None:
    """Return one mask, or nothing at all where it marks every sample.

    Args:
        held: The mask over the samples a crop keeps.

    Returns:
        mask: The mask, or None where every sample is true.
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
        budget: How many samples to read at once.

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
