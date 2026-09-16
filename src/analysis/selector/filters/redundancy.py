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
        track: The tile's observations on one time axis.
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
        owner, cells = track.owners[index], track.cells[index]
        filled = counter.observations_per_cell[owner]
        if int(np.count_nonzero(filled[cells] == 1)) >= gain:
            continue
        counter.release(owner, cells)
        spared = coverage_constraints(constraints, counter.cells_reached)
        # If the window can do without the observation
        if spared is not None:
            reached = spared
            kept.remove(index)
        else:
            counter.hold(owner, cells)
    return kept, reached
