"""What the whole record answers for, when a window cannot be asked for it."""

from __future__ import annotations

import math

import numpy as np

from analysis.selector.models.counter import Counter
from analysis.selector.models.filter import Filter
from analysis.selector.models.track import Track


def fresh_looks(track: Track, criteria: Filter) -> tuple[int, ...]:
    """Keep the best look a timeless instrument left on each part of the tile.

    Args:
        track: The tile's admissible observations on one time axis.
        criteria: The filter read against the tile, its timeless, bars and ranks.

    Returns:
        standing: Where they sit on the axis, oldest first, only new ground kept.
    """
    answering = {
        owner for owner, iid in enumerate(track.iids) if iid in criteria.timeless
    }
    unranked = [math.inf]
    ranked = sorted(
        (index for index, owner in enumerate(track.owners) if owner in answering),
        key=lambda index: criteria.ranks.get(track.observations[index].pdsid, unranked),
    )
    counter = Counter.empty(track.iids, track.grid.cells)
    held: list[int] = []
    for index in ranked:
        owner, cells = track.owners[index], track.cells[index]
        fresh = int(np.count_nonzero(counter.observations_per_cell[owner][cells] == 0))
        if fresh >= criteria.gain(track.iids[owner], cells.size):
            counter.hold(owner, cells)
            held.append(index)
    return tuple(sorted(held))
