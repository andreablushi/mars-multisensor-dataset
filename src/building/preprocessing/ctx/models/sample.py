"""One CTX scan cut to the feature it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.sample import Sample

# What the projection writes where the scan swept no ground.
BLANK = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class CtxSample(Sample):
    """The brightness one scan measured over one feature.

    Attributes:
        image: The brightness as lines by samples.
    """

    image: np.ndarray

    @property
    def blank(self) -> np.ndarray:
        """Return where the projection left no ground.

        Returns:
            Lines by samples, True at every pixel outside the corners the scan
            swept, worked out from the image rather than carried beside it.
        """
        return self.image == BLANK
