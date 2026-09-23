"""What a window has to hold from each instrument before it is worth keeping."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# One instrument answering a constraint: the sets that speak for it and its floor
Answer = tuple[tuple[int, ...], int]
# A constraint any one of its instruments can answer, and what a window is asked.
Constraints = list[list[Answer]]


@dataclass(slots=True)
class Filter:
    """What the instruments are asked for before a tile earns a place.

    Attributes:
        constraints: What a window meets all of, any one instrument answering each.
        admits: The pixels each instrument has to land on a tile to count, by iid.
        span_ls: How far round its year Mars may turn inside a window, in degrees.
        timeless: The instruments the ground answers for whenever they came.
        gain_share: The part of its own cells a look holds alone to be kept, by iid.
        least: The pixels each set has to land on the tile, by set.
        windowed: What a window is scored on, tightest constraint first.
        standing: What the whole record answers for, tightest first.
    """

    constraints: list[dict[str, float]]
    span_ls: float
    admits: dict[str, float] = field(default_factory=dict)
    timeless: list[str] = field(default_factory=list)
    gain_share: dict[str, float] = field(default_factory=dict)
    least: list[float] = field(default_factory=list)
    windowed: Constraints = field(default_factory=list)
    standing: Constraints = field(default_factory=list)

    def gain(self, iid: str, cells: int) -> int:
        """Return how many of its cells a look has to hold alone to be kept.

        Args:
            iid: The instrument the look belongs to.
            cells: How many of the tile's cells the look fills.

        Returns:
            gain: The cells, never fewer than one.
        """
        return max(1, math.ceil(self.gain_share.get(iid, 0.0) * cells))
