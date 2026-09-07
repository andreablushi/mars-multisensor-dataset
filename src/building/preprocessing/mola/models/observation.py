"""One MOLA grid cut to a box, on the projection its own label writes it in."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.relative_position import PolarGrid


@dataclass(frozen=True, slots=True)
class MolaObservation:
    """The height over one box, placed as the grid it was read from places it.

    Attributes:
        label: What the products it was read from say about it, merged.
        identifier: The grid it was read from, which every crop of it is
            stored under.
        topography: The height of the ground above the areoid in metres, as
            lines by samples, interpolated where no shot fell in the bin.
        down: What every line holds, its latitude in degrees on a cylindrical
            grid and its northing in the projection's metres on a cap.
        across: What every sample holds, its longitude or its easting, read the
            same way.
        polar: The pole the two are measured on, and None where they are the
            degrees a cylindrical grid places directly.
    """

    identifier: str
    label: dict[str, str]
    topography: np.ndarray
    down: np.ndarray
    across: np.ndarray
    polar: PolarGrid | None = None

    # Either projection is regular on both axes, so one axis places each side.
    separable = True
