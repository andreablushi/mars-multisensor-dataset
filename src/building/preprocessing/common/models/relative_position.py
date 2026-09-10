"""Where every sample of one observation sits, relative to its own feature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A polar grid: its centre longitude, whether north, and the sphere it is built on.
PolarGrid = tuple[float, bool, float]


@dataclass(frozen=True, slots=True)
class RelativePosition:
    """Where each sample of one observation sits on the feature it was kept for.

    Attributes:
        north: How far north of the feature centre, in degrees or a projection's own
            metres, one per line where the grid is separable and per sample where not.
        east: How far east of it, in the same unit, wrapped so the meridian is no
            jump, one per sample of a line where the grid is separable.
        separable: Whether the two hold one axis each, a line's north and a
            sample's east, rather than a value for every sample.
        polar: The grid the two are measured on, and None where they are the degrees
            every other placement holds.
    """

    north: np.ndarray
    east: np.ndarray
    separable: bool
    polar: PolarGrid | None = None

    @property
    def ground_sizes(self) -> tuple[int, ...]:
        """Return how many samples each ground axis holds.

        Returns:
            sizes: One count per ground axis, in the order those axes run.
        """
        if self.separable:
            return (self.north.size, self.east.size)
        return self.north.shape

    def offsets(self, taken: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
        """Return the northings and the eastings of the samples a cut keeps.

        Args:
            taken: Which of each ground axis to read, outermost first, and
                empty for all of them.

        Returns:
            north: The northings, one axis where the position is separable and a value
                per sample where it is not.
            east: The eastings, holding the same.
        """
        if not taken:
            return self.north, self.east
        if self.separable:
            return self.north[taken[0]], self.east[taken[1]]
        return self.north[taken], self.east[taken]
