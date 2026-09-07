"""Splitting a feature into cells about the size of a footprint, for speed alone."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
from shapely import STRtree, area, bounds, intersection, is_empty
from shapely.geometry.base import BaseGeometry

from analysis.coverage import configs
from analysis.coverage.models.grid import Grid
from analysis.coverage.models.region import FeatureRegion


def grid_over(region: FeatureRegion, shapes: Sequence[BaseGeometry]) -> Grid:
    """Size a grid to the footprints it will hold, and lay it over the feature.

    Args:
        region: The projected feature the cells cover.
        shapes: The projected footprints the grid will hold.

    Returns:
        grid: The grid, its side within the configured bounds.
    """
    west, south, east, north = region.shape.bounds
    boxes = bounds(np.asarray(shapes, dtype=object))
    spans = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    side = configs.MAX_UNION_CELLS
    # A set whose every footprint met the feature edge on has no span to size by
    if (spans > 0.0).any():
        typical = float(np.sqrt(np.median(spans[spans > 0.0])))
        wanted = round(math.sqrt((east - west) * (north - south)) / typical)
        side = int(min(max(wanted, configs.MIN_UNION_CELLS), configs.MAX_UNION_CELLS))
    return Grid(west=west, south=south, east=east, north=north, side=side)


def cells(
    grid: Grid, region: FeatureRegion, shapes: np.ndarray
) -> Iterator[tuple[BaseGeometry, float, np.ndarray]]:
    """Walk the cells that can hold ground, with what reaches each one.

    Args:
        grid: The grid laid over the feature.
        region: The projected feature, which bounds what a cell can hold.
        shapes: The projected footprints, in the order they are walked.

    Yields:
        cell: Each cell's rectangle, the ground it holds, and the shapes reaching it.
    """
    rectangles = grid.rectangles
    caps = area(intersection(rectangles, region.shape))
    index = STRtree(shapes)
    for rectangle, cap in zip(rectangles, caps, strict=True):
        if cap <= 0.0:
            continue
        reaching = np.sort(index.query(rectangle))
        if reaching.size:
            yield rectangle, float(cap), reaching


def clip(
    shapes: np.ndarray, reaching: np.ndarray, rectangle: BaseGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Cut the given shapes to one cell, dropping the ones that miss it.

    Args:
        shapes: Every projected footprint, indexed by the reaching indices.
        reaching: The indices of the shapes to cut.
        rectangle: The cell to cut them to.

    Returns:
        kept: The indices the box keeps.
        shapes: Their clipped shapes, one to each.
    """
    pieces = intersection(shapes[reaching], rectangle)
    kept = ~is_empty(pieces)
    return reaching[kept], pieces[kept]
