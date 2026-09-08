"""Splitting a feature into the fine cells its coverage is published on, for detail."""

from __future__ import annotations

import math

import numpy as np
from shapely import contains_xy, prepare
from shapely.geometry.base import BaseGeometry

from analysis.coverage.models.grid import Grid
from analysis.coverage.models.region import FeatureRegion

_NONE = np.empty(0, dtype=np.int64)

# How wide one block of the grid is, in kilometres, so large is not coarse
GRID_KM = 100

# A footprint under this share of a cell is given none, to credit no ground
MIN_CELL_SHARE = 0.5


def grid_over(region: FeatureRegion, grid_cells: int) -> Grid:
    """Give one feature a grid fine enough for the ground it covers.

    Args:
        region: The feature the footprints were cut to.
        grid_cells: How many cells one block of the grid holds along each axis.

    Returns:
        grid: The grid the feature is measured on.
    """
    west, south, east, north = region.shape.bounds
    span_km = math.sqrt((east - west) * (north - south)) / 1000.0
    return Grid(
        west=west,
        south=south,
        east=east,
        north=north,
        side=max(1, math.ceil(span_km / GRID_KM)) * grid_cells,
    )


def filled(grid: Grid, shape: BaseGeometry) -> np.ndarray:
    """Find the cells of the grid whose centre a shape covers.

    Args:
        grid: The grid the feature is measured on.
        shape: The projected shape to burn, already cut to the feature.

    Returns:
        cells: The indices of the cells it fills, in ascending order.
    """
    if shape.is_empty:
        return _NONE
    eastings, northings = grid.centres
    west, south, east, north = shape.bounds
    columns = np.nonzero((eastings >= west) & (eastings <= east))[0]
    rows = np.nonzero((northings >= south) & (northings <= north))[0]
    if columns.size and rows.size:
        across, down = np.meshgrid(eastings[columns], northings[rows])
        prepare(shape)
        inside = contains_xy(shape, across, down)
        if inside.any():
            line, crosswise = np.nonzero(inside)
            return rows[line] * grid.side + columns[crosswise]
    # A footprint holding no cell centre is given the one cell it sits in
    if shape.area >= grid.cell_area_m2 * MIN_CELL_SHARE:
        point = shape.representative_point()
        row = int(np.abs(northings - point.y).argmin())
        column = int(np.abs(eastings - point.x).argmin())
        return np.array([row * grid.side + column])
    return _NONE
