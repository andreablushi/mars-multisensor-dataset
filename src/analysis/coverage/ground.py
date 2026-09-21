"""Turning a count of grid cells into the ground it covers, and back again."""

from __future__ import annotations


def share(cells: float, cell_km2: float, area_km2: float) -> float:
    """Say how much of the ground a count of cells covers.

    Args:
        cells: How many cells of the grid are counted.
        cell_km2: How much ground one of them covers.
        area_km2: How much ground the whole of it holds.

    Returns:
        share: The share of that ground the cells cover, from nought to one.
    """
    return cells * cell_km2 / area_km2


def cells(share: float, area_km2: float, cell_km2: float) -> float:
    """Say how many cells a share of the ground comes to.

    Args:
        share: The share of the ground, from nought to one.
        area_km2: How much ground the whole of it holds.
        cell_km2: How much ground one cell covers.

    Returns:
        cells: The cells it comes to, which need not be a whole number of them.
    """
    return share * area_km2 / cell_km2
