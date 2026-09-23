"""The window worth the most, once Mars' own turning is priced against ground."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence

from analysis.coverage import ground
from analysis.selector.filters import redundancy
from analysis.selector.filters.coverage_constraints import coverage_constraints
from analysis.selector.models.counter import Counter
from analysis.selector.models.filter import Filter
from analysis.selector.models.survey import Survey
from analysis.selector.models.track import Track
from analysis.selector.models.window import Window

# How far Mars may turn for one more point of ground, in degrees; ten days at mean
LS_PER_PERCENT = 5.25

_PRICE_PER_DEGREE = 0.01 / LS_PER_PERCENT


def search(track: Track, criteria: Filter) -> Survey | None:
    """Search a timeline for the window the ground is best studied over.

    Args:
        track: The admissible observations on one time axis.
        criteria: The filter read against the tile, holding what it is asked.

    Returns:
        survey: The chosen window, or None when no window is worth keeping.
    """
    # What the filter asks of this tile, worked out once when it was read
    windowed, standing = criteria.windowed, criteria.standing
    # What time cannot change is asked of the whole record rather than a window
    if standing:
        whole = Counter.over(track, 0, len(track.observations) - 1)
        if coverage_constraints(standing, whole.cells_reached) is None:
            return None
    looked: list[list[int]] = []
    seen: list[dict[int, int]] = [{} for _ in track.iids]
    for index, owner in enumerate(track.owners):
        last = seen[owner]
        before: list[int] = []
        for cell in track.cells[index].tolist():
            before.append(last.get(cell, -1))
            last[cell] = index
        before.sort()
        looked.append(before)
    # Take the best window
    span_ls = criteria.span_ls
    picked: Window | None = None
    worth = float("-inf")
    # Loop over the observation as bound of the window
    for left in range(len(track.observations)):
        reached = [0] * len(track.iids)
        for right in range(left, len(track.observations)):
            arc = track.ls[right] - track.ls[left]
            if arc > span_ls:
                break
            fresh = bisect_left(looked[right], left)
            if not fresh:
                continue
            reached[track.owners[right]] += fresh
            counts = coverage_constraints(windowed, reached)
            if counts is None:
                continue  # the window does not hold what the filter asks
            paid = _scored(track, counts, arc)
            if paid > worth:
                days = track.times[right] - track.times[left]
                picked, worth = Window(left, right, days), paid
    if picked is None:
        return None
    # Clean up the record to only what is worth keeping, and report reached
    trimmed = redundancy.trimmed(track, picked, criteria)
    if trimmed is None:
        return None
    kept, standing, reached = trimmed
    return Survey(
        area_km2=track.grid.area_km2,
        start=track.observations[kept[0]].t_start,
        end=track.observations[kept[-1]].t_start,
        days=track.times[kept[-1]] - track.times[kept[0]],
        geo_mean=_scored(track, reached),
        kept=tuple(kept),
        standing=tuple(standing),
    )


def _scored(track: Track, counts: Sequence[int], arc: float = 0.0) -> float:
    """Score what a window reaches, less what the turning it spans cost it.

    Args:
        track: The observations on one time axis.
        counts: The cells each constraint reaches, one count per constraint.
        arc: How far Mars turns inside the window, charged against its ground.

    Returns:
        worth: The constraints rooted together as a share of the tile, less their arc.
    """
    rooted = math.prod(counts) ** (1.0 / len(counts))
    geo_mean = ground.share(rooted, track.grid.cell_km2, track.grid.area_km2)
    return geo_mean - _PRICE_PER_DEGREE * arc
