"""The stretch of time the search picked, and the search it came out of."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from common.analysis.coverage.models.coverage import SetCoverage
from common.analysis.selector.models import track as timeline
from common.analysis.selector.models.filter import Filter


@dataclass(frozen=True, slots=True)
class Survey:
    """The stretch of time one tile is best studied over.

    Attributes:
        area_km2: How much ground the tile covers.
        start: When the earliest observation inside it was taken.
        end: When the latest one was taken.
        days: How long it lasts.
        geo_mean: The insisted shares rooted together, as a share of the tile.
        kept: The observations it holds, as their places on the timeline, oldest first.
        standing: The observations kept from outside the window, oldest first.
    """

    area_km2: float
    start: datetime
    end: datetime
    days: float
    geo_mean: float
    kept: tuple[int, ...]
    standing: tuple[int, ...]

    @property
    def taken(self) -> tuple[int, ...]:
        """Name every observation the tile keeps, in time order.

        Returns:
            taken: The window's own observations and what came from outside it, oldest
                first.
        """
        return tuple(sorted(set(self.kept) | set(self.standing)))


@dataclass(frozen=True, slots=True)
class Study:
    """What the search found over one tile.

    Attributes:
        tile: The tile's name, such as "b123_c0456".
        criteria: What the tile was asked for.
        track: Its admissible observations on one time axis, or None where it
            holds nothing measurable.
        survey: The window it earned, or None where it earned none.
    """

    tile: str
    criteria: Filter
    track: timeline.Track | None
    survey: Survey | None

    @classmethod
    def over(cls, coverage: Sequence[SetCoverage], criteria: Filter) -> Study:
        """Search one tile under the filter.

        Args:
            coverage: The tile's instrument sets, in any order.
            criteria: Which instruments a window has to hold, and how much ground each.

        Returns:
            study: What the search found, the timeline it ran over and the window it
                earned.
        """
        # Imported here, since the algorithm hands back the survey defined above
        from common.analysis.selector import algorithm

        summary = coverage[0].summary
        settled, track = timeline.over(coverage, criteria)
        return cls(
            tile=summary.tile,
            criteria=settled,
            track=track,
            survey=algorithm.search(track, settled) if track else None,
        )
