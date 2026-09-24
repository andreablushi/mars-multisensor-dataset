"""What a window has to hold from each instrument before it is worth keeping."""

from __future__ import annotations

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
        redundant_share_threshold: The share of its cells a look shares with another
            past which the two are redundant, by iid.
        least: The pixels each set has to land on the tile, by set.
        windowed: What a window is scored on, tightest constraint first.
        standing: What the whole record answers for, tightest first.
    """

    constraints: list[dict[str, float]]
    span_ls: float
    admits: dict[str, float] = field(default_factory=dict)
    timeless: list[str] = field(default_factory=list)
    redundant_share_threshold: dict[str, float] = field(default_factory=dict)
    least: list[float] = field(default_factory=list)
    windowed: Constraints = field(default_factory=list)
    standing: Constraints = field(default_factory=list)
