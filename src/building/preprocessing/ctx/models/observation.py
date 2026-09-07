"""One CTX scan as it comes off disk, placed on its own grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from building.preprocessing.common.models.relative_position import PolarGrid


@dataclass(frozen=True, slots=True)
class CtxObservation:
    """One scan on the grid its label projects it onto.

    Attributes:
        label: What every product it was published as says about it, merged.
        identifier: The observation id.
        image: The TIFF the brightness was published as, read a window at a
            time rather than whole, since one feature keeps a small part of a
            scan that runs to gigabytes.
        down: What every line holds, its latitude in degrees on a cylindrical
            grid and its northing in the projection's metres on a polar one.
        across: What every sample holds, its longitude or its easting, read the
            same way.
        polar: The grid the two are measured on, and None where they are the
            degrees a cylindrical grid places directly.
    """

    identifier: str
    label: dict[str, str]
    image: Path
    down: np.ndarray
    across: np.ndarray
    polar: PolarGrid | None = None

    # Either projection is regular on both axes, so one axis places each side.
    separable = True
