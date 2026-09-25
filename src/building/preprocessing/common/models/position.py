"""Where every sample of one observation sits, on the grid it is measured on."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.maths.geodesy import PolarGrid


@dataclass(frozen=True, slots=True)
class Position:
    """Where each sample of one observation sits, absolute or from its tile centre.

    Attributes:
        north: The latitude or northing of every sample, one per line if separable.
        east: The longitude or easting, one per column if separable, else per sample.
        separable: Whether the two hold one ground axis each.
        grid: The polar grid the two are metres on, or None for degrees.
    """

    north: np.ndarray
    east: np.ndarray
    separable: bool
    grid: PolarGrid | None = None

    @property
    def sizes(self) -> tuple[int, ...]:
        """Return how many samples each ground axis holds.

        Returns:
            sizes: One count per ground axis, in the order those axes run.
        """
        if self.separable:
            return (self.north.size, self.east.size)
        return self.north.shape

    def crossed_part(self, taken: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Return the northings and eastings of the samples a cut takes, broadcastable.

        Args:
            taken: Which of each ground axis to read, outermost first.

        Returns:
            north: The northings, a column where separable, else one per sample.
            east: The eastings, a row where separable, else one per sample.
        """
        if self.separable:
            return self.north[taken[0]][:, None], self.east[taken[1]][None, :]
        return self.north[taken], self.east[taken]

    def dims_along(
        self, ground: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return which ground axes the northing and the easting each run along.

        Args:
            ground: The instrument's ground axes, in the order they run.

        Returns:
            north: The axes of the northing, one if separable, else every ground axis.
            east: The axes of the easting, holding the same.
        """
        if self.separable:
            return ground[:1], ground[1:]
        return ground, ground
