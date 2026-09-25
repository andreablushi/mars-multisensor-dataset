"""How much new ground each observation covers, accumulated cell by cell."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from shapely import Polygon, area, covers, intersection, is_empty, prepare, union_all
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry

from analysis.coverage.measurement.accumulation import coarse_split
from analysis.coverage.models.region import TileRegion

# How many observations a cell folds in before its union is rebuilt in one
UNION_CHUNK = 64

# A cell covered to within this share of what it could hold
SATURATION_TOLERANCE = 1e-12

# Grid an overlay is snapped to when exact arithmetic cannot node it
SNAP_GRID_M = 1e-6


def new_ground(
    region: TileRegion, shapes: Sequence[BaseGeometry], threads: int
) -> np.ndarray:
    """Measure the new ground every observation covers.

    Args:
        region: The projected tile the footprints are cut to.
        shapes: The projected footprints, in chronological order.
        threads: How many cells to accumulate at once, this job's share of the machine.

    Returns:
        ground: The ground in square metres each observation covered first.
    """
    footprints = np.asarray(shapes, dtype=object)
    grid = coarse_split.grid_over(region, footprints)
    ground = np.zeros(len(shapes), dtype=float)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for contributions in pool.map(
            lambda cell: new_ground_in_cell(footprints, *cell),
            coarse_split.reached_cells(grid, region, footprints),
        ):
            for index, added in contributions:
                ground[index] += added
    return ground


def new_ground_in_cell(
    footprints: np.ndarray, rectangle: BaseGeometry, cap: float, reaching: np.ndarray
) -> list[tuple[int, float]]:
    """Measure the new ground each observation reaching one cell covers inside it.

    Args:
        footprints: Every projected footprint, indexed by the reaching indices.
        rectangle: The cell being accumulated.
        cap: The ground in square metres the cell could ever hold.
        reaching: The indices of the observations reaching it, in order.

    Returns:
        contributions: Each observation adding ground, with the square metres added.
    """
    covered: BaseGeometry = Polygon()
    clipped: list[BaseGeometry] = []
    contributions: list[tuple[int, float]] = []
    saturation = cap * (1.0 - SATURATION_TOLERANCE)
    for start in range(0, reaching.size, UNION_CHUNK):
        chunk = reaching[start : start + UNION_CHUNK]
        pieces = intersection(footprints[chunk], rectangle)
        alive = ~is_empty(pieces)
        kept, pieces = chunk[alive], pieces[alive]
        if not kept.size:
            continue
        running = covered
        for index, piece in zip(kept, pieces, strict=True):
            # The first piece is its own union, which unioning it would round
            if running.is_empty:
                merged, added = piece, area(piece)
            elif covers(running, piece):
                continue
            else:
                merged = union_of([running, piece])
                added = merged.area - running.area
                if added <= running.area * SATURATION_TOLERANCE:
                    continue
            contributions.append((int(index), added))
            running = merged
            prepare(running)
        clipped.extend(pieces)
        covered = union_of(clipped)
        prepare(covered)
        if covered.area >= saturation:
            break
    return contributions


def union_of(shapes: Sequence[BaseGeometry]) -> BaseGeometry:
    """Union shapes, snapping them to a fine grid where exact arithmetic fails.

    Args:
        shapes: The shapes to union.

    Returns:
        union: The union of every shape.
    """
    try:
        return union_all(shapes)
    except GEOSException:
        # Exact arithmetic can fail on an overlay, which a fine grid settles
        return union_all(shapes, grid_size=SNAP_GRID_M)
