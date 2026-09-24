"""The coarse cells a tile is split into, each about a footprint wide, for speed."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from shapely import STRtree, area, bounds, intersection
from shapely.geometry.base import BaseGeometry

from analysis.coverage.models.grid import Grid
from analysis.coverage.models.region import TileRegion

# The union is kept per cell so each insert touches a small shape, not as a unit
MIN_UNION_CELLS = 4
MAX_UNION_CELLS = 32


def grid_over(region: TileRegion, shapes: np.ndarray) -> Grid:
    """Size a grid to the footprints it will hold, and lay it over the tile.

    Args:
        region: The projected tile the cells cover.
        shapes: The projected footprints the grid will hold.

    Returns:
        grid: The grid, its side within the configured bounds.
    """
    boxes = bounds(shapes)
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    box_areas = box_areas[box_areas > 0.0]
    side = MAX_UNION_CELLS
    # A set whose every footprint met the tile edge on has no span to size by
    if box_areas.size:
        typical = float(np.sqrt(np.median(box_areas)))
        wanted = round(region.span_m / typical)
        side = min(max(wanted, MIN_UNION_CELLS), MAX_UNION_CELLS)
    return Grid(*region.shape.bounds, side=side)


def reached_cells(
    grid: Grid, region: TileRegion, shapes: np.ndarray
) -> Iterator[tuple[BaseGeometry, float, np.ndarray]]:
    """Walk the cells that can hold ground, with what reaches each one.

    Args:
        grid: The grid laid over the tile.
        region: The projected tile, which bounds what a cell can hold.
        shapes: The projected footprints, in the order they are walked.

    Yields:
        cell: Each cell's rectangle, the ground it holds, and the shapes reaching it.
    """
    rectangles = grid.rectangles
    caps = area(intersection(rectangles, region.shape))
    tree = STRtree(shapes)
    for rectangle, cap in zip(rectangles, caps, strict=True):
        if cap <= 0.0:
            continue
        reaching = np.sort(tree.query(rectangle))
        if reaching.size:
            yield rectangle, float(cap), reaching
