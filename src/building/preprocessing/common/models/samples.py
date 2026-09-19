"""The samples of one observation, as the grid they were placed on holds them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.relative_position import PolarGrid


@dataclass(frozen=True, slots=True)
class Samples:
    """Where every sample of one observation sits, on the grid it was placed on.

    Attributes:
        down: The latitude of every line in degrees, or its northing in the
            metres of `grid`, one per sample where the two are not separable.
        across: The longitude of every sample, or its easting, holding the same.
        separable: Whether those two hold one axis each rather than a value for
            every sample.
        grid: The grid the two are measured on, and None where they are degrees.
    """

    down: np.ndarray
    across: np.ndarray
    separable: bool
    grid: PolarGrid | None

    @property
    def sizes(self) -> tuple[int, ...]:
        """Return how many samples each ground axis holds.

        Returns:
            sizes: One count per ground axis, in the order those axes run.
        """
        if self.separable:
            return (self.down.size, self.across.size)
        return self.down.shape
