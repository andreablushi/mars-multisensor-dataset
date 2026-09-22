"""Which samples of one observation the box of its tile keeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Cut:
    """What one tile's box keeps of one observation, before it is placed.

    Attributes:
        bounds: The samples to keep of each ground axis, outermost first.
        inside: Which kept samples truly fall in the box, or None for all.
        separable: Whether it keeps one axis each rather than a value per sample.
    """

    bounds: tuple[np.ndarray, ...]
    inside: np.ndarray | None
    separable: bool
