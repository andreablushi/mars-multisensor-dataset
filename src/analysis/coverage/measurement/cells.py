"""The cells a tile's coverage is counted on, and those a shape fills."""

from __future__ import annotations

import math

import numpy as np
from shapely import contains_xy, prepare
from shapely.geometry.base import BaseGeometry

from analysis.coverage.models.cell_grid import CellGrid
from analysis.coverage.models.region import TileRegion
from common.maths import physics

_NONE = np.empty(0, dtype=np.int64)

# A footprint under this share of a cell is given none, to credit no ground
MIN_CELL_SHARE = 0.5


def cell_grid(region: TileRegion, cell_km: float) -> CellGrid:
    """Give one tile a grid fine enough for the ground it covers.

    Args:
        region: The tile the footprints were cut to.
        cell_km: The side one cell is sized to, in kilometres.

    Returns:
        grid: The grid the tile is measured on.
    """
    span_km = region.span_m / physics.METRES_PER_KM
    return CellGrid(*region.laea.bounds, side=max(1, math.ceil(span_km / cell_km)))


def filled_cells(grid: CellGrid, shape: BaseGeometry) -> np.ndarray:
    """Find the cells of the grid whose centre a shape covers.

    Args:
        grid: The grid the tile is measured on.
        shape: The projected shape to burn, already cut to the tile.

    Returns:
        cells: The indices of the cells it fills, in ascending order.
    """
    eastings, northings = grid.centres
    west, south, east, north = shape.bounds
    columns = np.nonzero((eastings >= west) & (eastings <= east))[0]
    rows = np.nonzero((northings >= south) & (northings <= north))[0]
    if columns.size and rows.size:
        across, down = np.meshgrid(eastings[columns], northings[rows])
        prepare(shape)
        inside = contains_xy(shape, across, down)
        if inside.any():
            hit_rows, hit_columns = np.nonzero(inside)
            return rows[hit_rows] * grid.side + columns[hit_columns]
    # A footprint holding no cell centre is given the one cell it sits in
    if shape.area >= grid.cell_area_m2 * MIN_CELL_SHARE:
        point = shape.representative_point()
        row = int(np.abs(northings - point.y).argmin())
        column = int(np.abs(eastings - point.x).argmin())
        return np.array([row * grid.side + column])
    return _NONE
