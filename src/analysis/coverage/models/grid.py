"""The grid a coverage measurement counts cells on, over one feature's box."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from shapely import box


@dataclass(frozen=True)
class Grid:
    """A regular grid of cells covering one feature's projected bounding box.

    Attributes:
        west: The westernmost easting the grid spans, in metres.
        south: The southernmost northing it spans, in metres.
        east: The easternmost easting it spans, in metres.
        north: The northernmost northing it spans, in metres.
        side: How many cells it holds along each axis.
    """

    west: float
    south: float
    east: float
    north: float
    side: int

    @cached_property
    def cell_area_m2(self) -> float:
        """Return how much ground one cell of the grid covers.

        Returns:
            area: The area of one cell in square metres.
        """
        return (self.east - self.west) * (self.north - self.south) / self.side**2

    @cached_property
    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Return where the centre of every cell falls, along each axis.

        Returns:
            eastings: The cell centre eastings in metres.
            northings: The cell centre northings in metres.
        """
        steps = np.arange(self.side) + 0.5
        return (
            self.west + steps * (self.east - self.west) / self.side,
            self.south + steps * (self.north - self.south) / self.side,
        )

    @cached_property
    def rectangles(self) -> np.ndarray:
        """Return every cell of the grid as a rectangle.

        Returns:
            cells: One box per cell, walked row by row from the south west corner.
        """
        step_x = (self.east - self.west) / self.side
        step_y = (self.north - self.south) / self.side
        return np.asarray(
            [
                box(
                    self.west + column * step_x,
                    self.south + row * step_y,
                    self.west + (column + 1) * step_x,
                    self.south + (row + 1) * step_y,
                )
                for row in range(self.side)
                for column in range(self.side)
            ],
            dtype=object,
        )
