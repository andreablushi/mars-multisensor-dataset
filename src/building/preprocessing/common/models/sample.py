"""What every instrument's crop holds, whatever its own arrays are."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.relative_position import RelativePosition


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    """One observation cut to its feature, in the shape its instrument publishes.

    Attributes:
        identifier: What the instrument was asked for, its observation or tile.
        position: Where the samples that are left sit, in degrees from the
            feature's own centre.
        inside: Which of them truly falls in the feature's box, or None where
            every one of them does.
    """

    identifier: str
    position: RelativePosition
    inside: np.ndarray | None = None
