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
            feature's own centre or in the metres of the projection they
            were placed on.
        label: What every product the observation was published as says about
            it, merged into one.
        inside: Which of them truly falls in the feature's box, or None where
            every one of them does.
        valid: Which of them is a measurement rather than a filled cell, or
            None where every one of them is.
    """

    identifier: str
    position: RelativePosition
    label: dict[str, str]
    inside: np.ndarray | None = None
    valid: np.ndarray | None = None

    @property
    def measured(self) -> np.ndarray:
        """Return which samples are both in the feature's box and measurements.

        Returns:
            measured: One flag per sample over the ground axes, the two masks
                rooted together and every sample marked where neither says less.
        """
        held = np.ones(self.position.ground_sizes, dtype=bool)
        for mask in (self.inside, self.valid):
            if mask is not None:
                held = held & mask
        return held
