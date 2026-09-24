"""The window worth the most, once Mars' own turning is priced against ground."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence

from analysis.selector.filters import redundancy
from analysis.selector.filters.coverage_constraints import cells_per_constraint
from analysis.selector.models.counter import Counter
from analysis.selector.models.filter import Filter
from analysis.selector.models.survey import Survey
from analysis.selector.models.track import Track


def search(track: Track, criteria: Filter) -> Survey | None:
    """Search a timeline for the window the ground is best studied over.

    Args:
        track: The admissible observations on one time axis.
        criteria: The filter the tile is searched under.

    Returns:
        survey: The chosen window, or None when no window is worth keeping.
    """
    count = len(track.observations)
    # What the filter asks of this tile, worked out once when it was read
    windowed, standing = track.windowed, track.standing
    # What time cannot change is asked of the whole record rather than a window
    if standing:
        whole = Counter.over(track, range(count))
        if cells_per_constraint(standing, whole.cells_reached) is None:
            return None
    # For each observation, where its set last saw each of its cells, sorted
    last_seen_before: list[list[int]] = []
    last_seen: list[dict[int, int]] = [{} for _ in track.iids]
    for index, owner in enumerate(track.owners):
        seen_by_owner = last_seen[owner]
        before: list[int] = []
        for cell in track.cells[index].tolist():
            before.append(seen_by_owner.get(cell, -1))
            seen_by_owner[cell] = index
        before.sort()
        last_seen_before.append(before)
    # Take the best window
    best: tuple[int, int] | None = None
    best_score = float("-inf")
    # Loop over the observations as bounds of the window
    for left in range(count):
        reached = [0] * len(track.iids)
        for right in range(left, count):
            arc = track.ls[right] - track.ls[left]
            if arc > criteria.span_ls:
                break
            fresh = bisect_left(last_seen_before[right], left)
            if not fresh:
                continue
            reached[track.owners[right]] += fresh
            counts = cells_per_constraint(windowed, reached)
            if counts is None:
                continue  # the window does not hold what the filter asks
            score = geo_mean(track, counts) - criteria.ls_price * arc
            if score > best_score:
                best, best_score = (left, right), score
    if best is None:
        return None
    # Clean up the record to only what is worth keeping, and report reached
    kept, kept_standing, reached = redundancy.trimmed(track, *best, criteria)
    return Survey(
        start=track.observations[kept[0]].t_start,
        end=track.observations[kept[-1]].t_start,
        days=track.times[kept[-1]] - track.times[kept[0]],
        geo_mean=geo_mean(track, reached),
        taken=tuple(sorted(kept + kept_standing)),
        standing=frozenset(kept_standing),
    )


def geo_mean(track: Track, counts: Sequence[int]) -> float:
    """Root together the cells each constraint reaches, as a share of the tile.

    Args:
        track: The observations on one time axis.
        counts: The cells each constraint reaches, one count per constraint.

    Returns:
        share: Their geometric mean, as a share of the tile's ground.
    """
    rooted = math.prod(counts) ** (1.0 / len(counts))
    return rooted * track.grid.cell_km2 / track.grid.area_km2
