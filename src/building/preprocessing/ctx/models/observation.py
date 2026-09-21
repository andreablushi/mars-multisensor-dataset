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
        image: The brightness TIFF, read a window at a time.
        down: The latitude or northing of every line.
        across: The longitude or easting of every sample.
        polar: The grid the two are measured on, or None for degrees.
    """

    identifier: str
    label: dict[str, str]
    image: Path
    down: np.ndarray
    across: np.ndarray
    polar: PolarGrid | None = None

    # Either projection is regular on both axes, so one axis places each side.
    separable = True
