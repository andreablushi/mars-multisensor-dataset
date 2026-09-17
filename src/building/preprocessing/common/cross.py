"""Crossing the samples of one cut in blocks, since a grid can run to gigabytes."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from building.preprocessing.common.models.samples import Samples
from shared.maths import geodesy

# How many samples of a cut are crossed at once.
BLOCK = 4_000_000


def blocked(sizes: tuple[int, ...]) -> Iterator[slice]:
    """Walk the lines of one cut in blocks of about as many samples as fit.

    Args:
        sizes: How many samples each ground axis holds.

    Yields:
        block: Which lines of the cut to read.
    """
    reach = max(1, BLOCK // max(1, int(np.prod(sizes[1:], dtype=int))))
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
