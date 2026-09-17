"""Cutting the arrays of one observation down to the samples a box keeps."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from shared.maths import geodesy


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


def sample_sizes(
    down: np.ndarray, across: np.ndarray, separable: bool
) -> tuple[int, ...]:
    """Return how many samples each ground axis of one cut holds.

    Args:
        down: What places every line of it, or every sample where the two are
            not separable.
        across: What places every sample of a line, holding the same.
        separable: Whether those two hold one axis each.

    Returns:
        sizes: One count per ground axis, in the order those axes run.
    """
    return (down.size, across.size) if separable else down.shape


def sampled(
    down: np.ndarray,
    across: np.ndarray,
    separable: bool,
    grid: tuple[float, bool, float] | None,
    sizes: tuple[int, ...],
    block_samples: int,
) -> Iterator[tuple[slice, np.ndarray, np.ndarray]]:
    """Walk the samples of one cut in blocks, as the longitude and latitude of each.

    Args:
        down: The latitude of every line in degrees, or its northing in the
            metres of `grid`, one per sample where the two are not separable.
        across: The longitude of every sample, or its easting, holding the same.
        separable: Whether those two hold one axis each rather than a value for
            every sample.
        grid: The grid the two are measured on, and None where they are degrees.
        sizes: How many samples each ground axis holds.
        block_samples: How many samples to read at once, so a grid running to
            gigabytes is never crossed whole.

    Yields:
        block: Which lines of the cut this hands back.
        longitude: Their longitudes in degrees, one row per line of the block.
        latitude: Their latitudes, holding the same.
    """
    reach = max(1, block_samples // max(1, int(np.prod(sizes[1:], dtype=int))))
    for start in range(0, sizes[0], reach):
        block = slice(start, start + reach)
        if separable:
            first, second = down[block][:, None], across[None, :]
        else:
            first, second = down[block], across[block]
        if grid is None:
            yield block, second, first
            continue
        lon, lat = geodesy.stereographic_inverse(second, first, *grid)
        yield block, lon, lat
