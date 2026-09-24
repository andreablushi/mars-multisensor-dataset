"""The count of what a sliding window holds, kept without recounting each step."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from analysis.selector.models.track import Track


@dataclass(slots=True)
class Counter:
    """What one window holds, kept true as the window slides along the axis.

    Attributes:
        observations_per_cell: One row of observation counts per cell, per set.
        cells_reached: How many cells of the tile each set reaches.
    """

    observations_per_cell: np.ndarray
    cells_reached: list[int]

    @classmethod
    def over(cls, track: Track, indices: Iterable[int]) -> Counter:
        """Count afresh everything some observations of the axis hold.

        Args:
            track: The tile's observations on one time axis.
            indices: The observations to count, as indices into the track.

        Returns:
            counter: The counter, counting those observations.
        """
        counter = cls(
            observations_per_cell=np.zeros(
                (len(track.iids), track.grid.cell_count), dtype=np.int32
            ),
            cells_reached=[0] * len(track.iids),
        )
        for index in indices:
            counter.hold(track.owners[index], track.cells[index])
        return counter

    def hold(self, owner: int, cells: np.ndarray) -> None:
        """Take one more observation into the window.

        Args:
            owner: The instrument set the observation belongs to.
            cells: The tile's cells it fills, each of them named once.
        """
        counts = self.observations_per_cell[owner]
        self.cells_reached[owner] += cells.size - int(np.count_nonzero(counts[cells]))
        counts[cells] += 1

    def release(self, owner: int, cells: np.ndarray) -> None:
        """Drop one observation back out of the window.

        Args:
            owner: The instrument set the observation belongs to.
            cells: The tile's cells it fills, each of them named once.
        """
        counts = self.observations_per_cell[owner]
        counts[cells] -= 1
        # Ground nothing else left in the window reaches
        self.cells_reached[owner] -= cells.size - int(np.count_nonzero(counts[cells]))
