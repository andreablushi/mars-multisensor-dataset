"""One observation cut down to the ground its own feature covers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.relative_position import RelativePosition


@dataclass(frozen=True, slots=True)
class Crop[Sample]:
    """One observation cut to its feature, in the shape its instrument publishes.

    Attributes:
        sample: The instrument's own sample, every array of it cut to the box.
        position: Where the samples that are left sit, cut the same way.
        inside: Which of them truly falls in the box, or None where every one
            of them does.
    """

    sample: Sample
    position: RelativePosition
    inside: np.ndarray | None
