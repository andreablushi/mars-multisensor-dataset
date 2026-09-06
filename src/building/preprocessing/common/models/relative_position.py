"""Where every sample of one observation sits, relative to its own feature."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The polar stereographic grid a placement may be measured on: the longitude it
# is centred on, whether that centre is the north pole rather than the south,
# and the sphere it is built on in metres. A grid like this is regular in metres
# rather than in degrees, so a placement on it keeps one axis per side only
# while it stays in the projection's own metres.
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
    def ground_axes(self) -> int:
        """Return how many axes of ground the position places.

        Returns:
            The two a separable grid crosses, and otherwise the axes the
            offsets are already held over.
        """
        return 2 if self.separable else self.north.ndim
