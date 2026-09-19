"""How much new ground each observation covers, accumulated cell by cell."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from shapely import Polygon, area, covers, prepare, union_all
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry

from common.analysis.coverage.measurement.accumulation import coarse_split
from common.analysis.coverage.models.region import TileRegion

# How many observations a sector folds in before its union is rebuilt in one
UNION_CHUNK = 64

# A sector covered to within this share of what it could hold
SATURATION_TOLERANCE = 1e-12

# Grid an overlay is snapped to when exact arithmetic cannot node it.
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
    indexed = np.asarray(shapes, dtype=object)
    grid = coarse_split.grid_over(region, indexed)

    def cell_contributions(
        rectangle: BaseGeometry, cap: float, reaching: np.ndarray
    ) -> list[tuple[int, float]]:
        """Accumulate one cell and report what it contributes to each observation.

        Args:
            rectangle: The cell being accumulated.
            cap: The ground in square metres the cell could ever hold.
            reaching: The indices of the observations reaching it, in order.

        Returns:
            ground: The ground in square metres this cell saw each observation cover
                first.
        """
        covered: BaseGeometry = Polygon()
        arrived: list[BaseGeometry] = []
        share: list[tuple[int, float]] = []
        limit = cap * (1.0 - SATURATION_TOLERANCE)
        for start in range(0, reaching.size, UNION_CHUNK):
            indices, pieces = coarse_split.clip(
                indexed, reaching[start : start + UNION_CHUNK], rectangle
            )
            if not indices.size:
                continue
            running = covered
            for index, piece in zip(indices, pieces, strict=True):
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
                share.append((int(index), added))
                running = merged
                prepare(running)
            arrived.extend(pieces)
            covered = union_of(arrived)
            prepare(covered)
            if covered.area >= limit:
                break
        return share

    fresh = np.zeros(len(shapes), dtype=float)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for share in pool.map(
            lambda cell: cell_contributions(*cell),
            coarse_split.cells(grid, region, indexed),
        ):
            for index, added in share:
                fresh[index] += added
    return fresh


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
