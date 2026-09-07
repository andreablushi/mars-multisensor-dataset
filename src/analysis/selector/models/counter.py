"""Counting what a sliding window holds, without recounting it each step."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from analysis.selector.models.track import Track


@dataclass(slots=True)
class Counter:
    """What one window holds, kept true as the window slides along the axis.

    Attributes:
        observations_per_cell: The window's observations filling each cell, per set,
            as one row of counts per set.
        cells_reached: How many cells of the feature each set reaches.
    """

    observations_per_cell: np.ndarray
    cells_reached: list[int]

    @classmethod
    def empty(cls, iids: Sequence[str], grid_cells: int) -> Counter:
        """Open a counter on a window holding nothing at all.

        Args:
            iids: The instrument each set of the feature belongs to, by set.
            grid_cells: How many cells the feature holds.

        Returns:
            counter: The counter, counting nothing.
        """
        return cls(
            observations_per_cell=np.zeros((len(iids), grid_cells), dtype=np.int32),
            cells_reached=[0] * len(iids),
        )

    @classmethod
    def over(cls, track: Track, first: int, last: int) -> Counter:
        """Count afresh everything one stretch of the axis holds.

        Args:
            track: The feature's observations on one time axis.
            first: The index of the earliest observation the stretch holds.
            last: The index of the latest one.

        Returns:
            counter: The counter, counting that stretch.
        """
        counter = cls.empty(track.iids, track.grid.cells)
        for index in range(first, last + 1):
            counter.hold(track.owners[index], track.cells[index])
        return counter

    def hold(self, owner: int, cells: np.ndarray) -> None:
        """Take one more observation into the window.

        Args:
            owner: The instrument set the observation belongs to.
            cells: The feature's cells it fills, each of them named once.
        """
        filled = self.observations_per_cell[owner]
        held = filled[cells]
        self.cells_reached[owner] += cells.size - int(np.count_nonzero(held))
        held += 1
        filled[cells] = held

    def release(self, owner: int, cells: np.ndarray) -> None:
        """Drop the oldest observation back out of the window.

        Args:
            owner: The instrument set the observation belongs to.
            cells: The feature's cells it fills, each of them named once.
        """
        filled = self.observations_per_cell[owner]
        left = filled[cells]
        left -= 1
        filled[cells] = left
        # ground nothing else left in the window reaches
        self.cells_reached[owner] -= cells.size - int(np.count_nonzero(left))
