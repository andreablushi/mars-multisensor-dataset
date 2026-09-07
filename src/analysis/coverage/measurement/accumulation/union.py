"""How much new ground each observation covers, accumulated cell by cell."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from shapely import Polygon, area, covers, prepare, union_all
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry

from analysis.coverage import configs
from analysis.coverage.measurement.accumulation import coarse_split
from analysis.coverage.models.region import FeatureRegion


def new_ground(
    region: FeatureRegion, shapes: Sequence[BaseGeometry], threads: int
) -> np.ndarray:
    """Measure the new ground every observation covers.

    Args:
        region: The projected feature the footprints are cut to.
        shapes: The projected footprints, in chronological order.
        threads: How many cells to accumulate at once, this job's share of the machine.

    Returns:
        ground: The ground in square metres each observation covered first.
    """
    indexed = np.asarray(shapes, dtype=object)
    grid = coarse_split.grid_over(region, indexed)
    fresh = np.zeros(len(shapes), dtype=float)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for share in pool.map(
            lambda cell: _cell_contributions(indexed, *cell),
            coarse_split.cells(grid, region, indexed),
        ):
            for index, added in share:
                fresh[index] += added
    return fresh


def _cell_contributions(
    shapes: np.ndarray,
    rectangle: BaseGeometry,
    cap: float,
    reaching: np.ndarray,
) -> list[tuple[int, float]]:
    """Accumulate one cell and report what it contributes to each observation.

    Args:
        shapes: Every projected footprint, indexed by the reaching indices.
        rectangle: The cell being accumulated.
        cap: The ground in square metres the cell could ever hold.
        reaching: The indices of the observations reaching it, in order.

    Returns:
        ground: The ground in square metres this cell saw each observation cover first.
    """
    covered: BaseGeometry = Polygon()
    arrived: list[BaseGeometry] = []
    share: list[tuple[int, float]] = []
    limit = cap * (1.0 - configs.SATURATION_TOLERANCE)
    for start in range(0, reaching.size, configs.UNION_CHUNK):
        indices, pieces = coarse_split.clip(
            shapes, reaching[start : start + configs.UNION_CHUNK], rectangle
        )
        if not indices.size:
            continue
        _record_first_cover(indices, pieces, covered, share)
        arrived.extend(pieces)
        try:
            covered = union_all(arrived)
        except GEOSException:
            # Exact arithmetic can fail on an overlay, which a fine grid settles
            covered = union_all(arrived, grid_size=configs.SNAP_GRID_M)
        prepare(covered)
        if covered.area >= limit:
            break
    return share


def _record_first_cover(
    indices: np.ndarray,
    pieces: np.ndarray,
    covered: BaseGeometry,
    share: list[tuple[int, float]],
) -> None:
    """Record what each footprint in one chunk newly covers.

    Args:
        indices: The observation index of every piece, in order.
        pieces: The footprints clipped to the cell, in the same order.
        covered: The cell's union before this chunk, empty for the first.
        share: The cell's contributions so far, appended to in place.
    """
    running = covered
    for index, piece in zip(indices, pieces, strict=True):
        # The first piece is its own union, which unioning it would round
        if running.is_empty:
            share.append((int(index), area(piece)))
            running = piece
            prepare(running)
            continue
        if covers(running, piece):
            continue
        try:
            merged = union_all([running, piece])
        except GEOSException:
            # Exact arithmetic can fail on an overlay, which a fine grid settles
            merged = union_all([running, piece], grid_size=configs.SNAP_GRID_M)
        added = merged.area - running.area
        if added <= running.area * configs.SATURATION_TOLERANCE:
            continue
        share.append((int(index), added))
        running = merged
        prepare(running)
