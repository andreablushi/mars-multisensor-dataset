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
        north: How far north of the feature centre, in degrees, or in the
            projection's own metres where one is set. One per line where the
            grid is separable, and one per sample where it is not.
        east: How far east of it, in the same unit, wrapped so the meridian is
            no jump where that unit is degrees. One per sample of a line where
            the grid is separable, and otherwise shaped as `north` is.
        separable: Whether the two hold one axis each, a line's north and a
            sample's east, rather than a value for every sample.
        polar: The grid the two are measured on, and None where they are the
            degrees every other placement holds. Its metres are the
            projection's own and not the ground's, the two differing by a scale
            that rises away from the pole, so a distance is measured off the
            degrees it inverts to rather than read from the offsets.
    """

    north: np.ndarray
    east: np.ndarray
    separable: bool
    polar: PolarGrid | None = None

    @property
    def ground_sizes(self) -> tuple[int, ...]:
        """Return how many samples each ground axis holds.

        Returns:
            One count per ground axis, in the order those axes run.
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
            The northings and the eastings, holding one axis each where the
            position is separable and a value per sample where it is not.
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
            The axes of the northing and those of the easting, one each where
            the position is separable and every ground axis where it is not.
        """
        if self.separable:
            return ground[:1], ground[1:]
        return ground, ground
