"""One tile's observations on one time axis, and what the filter asks of them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.coverage.models.coverage import Event
from analysis.selector.models.filter import Constraints
from analysis.selector.models.search_grid import SearchGrid

# The observations offered to a tile, the set each belongs to, and the cells each fills
Offered = list[tuple[Event, int, list[int]]]


@dataclass(frozen=True, slots=True)
class Track:
    """One tile's observations, from every instrument set, in time order.

    Attributes:
        observations: The observations the search may pick from, oldest first.
        times: When each of them started, in days, which is what a span is measured in.
        ls: The solar longitude at which each was taken, in degrees.
        owners: The instrument set each belongs to, as its index into labels.
        cells: The tile's cells each fills, in the same order, each named once.
        labels: The name of each set, in the order owners index them.
        iids: The instrument each set belongs to, in the same order.
        grid: The grid the tile is searched over.
        refused: The observations left off the axis, with their sets and cells.
        min_pixels: The pixels each set has to land on the tile, by set.
        windowed: What a window is scored on, tightest constraint first.
        standing: What the whole record answers for, tightest first.
    """

    observations: list[Event]
    times: list[float]
    ls: list[float]
    owners: list[int]
    cells: list[np.ndarray]
    labels: list[str]
    iids: list[str]
    grid: SearchGrid
    refused: Offered
    min_pixels: list[float]
    windowed: Constraints
    standing: Constraints
