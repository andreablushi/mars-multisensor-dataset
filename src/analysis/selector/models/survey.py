"""The stretch of time the search picked for one tile."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Survey:
    """The stretch of time one tile is best studied over.

    Attributes:
        start: When the earliest observation inside it was taken.
        end: When the latest one was taken.
        days: How long it lasts.
        geo_mean: The insisted shares rooted together, as a share of the tile.
        taken: Every observation the tile keeps, as its place on the timeline,
            oldest first.
        standing: The timeless observations among them, kept whenever they came.
    """

    start: datetime
    end: datetime
    days: float
    geo_mean: float
    taken: tuple[int, ...]
    standing: frozenset[int]
