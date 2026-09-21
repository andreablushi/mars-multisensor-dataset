"""Which samples of one observation the box of its tile keeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Cut:
    """What one tile's box keeps of one observation, before it is placed.

    Attributes:
        bounds: The samples to keep of each ground axis, outermost first.
        inside: Which of the samples that survives the cut truly falls in the
            box, or None where every one of them does.
        separable: Whether what it keeps holds one axis each rather than a value
            for every sample.
    """

    bounds: tuple[np.ndarray, ...]
    inside: np.ndarray | None
    separable: bool
