"""One CTX scan cut to the tile it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mars_multisensor_dataset.building.preprocessing.common.models.sample import Sample

# What the projection writes where the scan swept no ground.
BLANK = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class CtxSample(Sample):
    """The brightness one scan measured over one tile.

    Attributes:
        image: The brightness as lines by samples.
        encoded: The arrays the image is stored as instead of float16, or None.
        encoding: What reads those arrays back, or None likewise.
    """

    image: np.ndarray
    encoded: dict[str, np.ndarray] | None = None
    encoding: dict[str, object] | None = None
