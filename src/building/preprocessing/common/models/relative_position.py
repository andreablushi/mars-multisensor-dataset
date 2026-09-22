"""Where every sample of one observation sits, relative to its own tile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.maths.geodesy import PolarGrid


@dataclass(frozen=True, slots=True)
class RelativePosition:
    """Where each sample of one observation sits on the tile it was kept for.

    Attributes:
        north: How far north of the tile centre, in degrees or metres.
        east: How far east of it, in the same unit, wrapped at the meridian.
        separable: Whether the two hold one axis each.
        polar: The grid the two are measured on, or None for degrees.
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
            taken: Which of each ground axis to read, outermost first, empty for all.

        Returns:
            north: The northings, one axis if separable, else one per sample.
            east: The eastings, holding the same.
        """
        if not taken:
            return self.north, self.east
        if self.separable:
            return self.north[taken[0]], self.east[taken[1]]
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
