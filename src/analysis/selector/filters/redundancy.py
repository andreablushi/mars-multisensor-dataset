"""Which observations a window can do without, and what the rest still reach."""

from __future__ import annotations

import numpy as np

from analysis.selector.filters.coverage_constraints import coverage_constraints
from analysis.selector.models.counter import Counter
from analysis.selector.models.filter import Constraints
from analysis.selector.models.track import Track
from analysis.selector.models.window import Window


def trimmed(
    track: Track, window: Window, constraints: Constraints, gain: int
) -> tuple[list[int], list[int]]:
    """Drop the observations a window does not need, oldest first.

    Args:
        track: The feature's observations on one time axis.
        window: The window they are counted inside.
        constraints: The cells each instrument insisted on has to reach.
        gain: The cells an observation has to bring that its own set does not reach.

    Returns:
        kept: The observations worth keeping, oldest first.
        reached: The cells each constraint still reaches once the rest are gone.
    """
    # List of the observations that are kept
    kept = list(range(window.first, window.last + 1))
    # Count what the window holds in cells
    counter = Counter.over(track, window.first, window.last)
    reached = coverage_constraints(constraints, counter.cells_reached)
    # Try to drop each observation, oldest first, and keep the rest
    for index in list(kept):
        spared = _without(track, counter, constraints, index, gain)
        # If the window can do without the observation
        if spared is not None:
            reached = spared
            kept.remove(index)
    return kept, reached


def _without(
    track: Track, counter: Counter, constraints: Constraints, index: int, gain: int
) -> list[int] | None:
    """Take one observation out of a window, unless the window needs it.

    Args:
        track: The feature's observations on one time axis.
        counter: What the window holds, which the observation is taken out of.
        constraints: The cells each instrument insisted on has to reach.
        index: The observation to try the window without.
        gain: The cells it has to bring that its own set does not already reach.

    Returns:
        counts: The cells each constraint reaches without it, or None when it is needed.
    """
    owner, cells = track.owners[index], track.cells[index]
    filled = counter.observations_per_cell[owner]
    if int(np.count_nonzero(filled[cells] == 1)) >= gain:
        return None
    counter.release(owner, cells)
    counts = coverage_constraints(constraints, counter.cells_reached)
    if counts is None:
        counter.hold(owner, cells)
        return None
    return counts
