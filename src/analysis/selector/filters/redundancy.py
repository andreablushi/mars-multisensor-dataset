"""Which observations a window can do without, and what the rest still reach."""

from __future__ import annotations

import numpy as np

from analysis.selector.filters.coverage_constraints import coverage_constraints
from analysis.selector.models.counter import Counter
from analysis.selector.models.filter import Filter
from analysis.selector.models.track import Track
from analysis.selector.models.window import Window


def trimmed(
    track: Track, window: Window, criteria: Filter
) -> tuple[list[int], list[int]]:
    """Drop the observations a window does not need, oldest first.

    Args:
        track: The tile's observations on one time axis.
        window: The window they are counted inside.
        criteria: The filter read against the tile, its constraints and its bar.

    Returns:
        kept: The observations worth keeping, oldest first.
        reached: The cells each constraint still reaches once the rest are gone.
    """
    # List of the observations that are kept
    kept = list(range(window.first, window.last + 1))
    # Count what the window holds in cells
    counter = Counter.over(track, window.first, window.last)
    reached = coverage_constraints(criteria.windowed, counter.cells_reached)
    # Try to drop each observation, oldest first, and keep the rest
    for index in list(kept):
        owner, cells = track.owners[index], track.cells[index]
        filled = counter.observations_per_cell[owner]
        alone = int(np.count_nonzero(filled[cells] == 1))
        if alone >= criteria.gain(track.iids[owner], cells.size):
            continue
        counter.release(owner, cells)
        spared = coverage_constraints(criteria.windowed, counter.cells_reached)
        # If the window can do without the observation
        if spared is not None:
            reached = spared
            kept.remove(index)
        else:
            counter.hold(owner, cells)
    return kept, reached
