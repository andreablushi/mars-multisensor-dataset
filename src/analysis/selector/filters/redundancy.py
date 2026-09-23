"""Which observations a tile can do without, and what the rest still reach."""

from __future__ import annotations

import numpy as np

from analysis import configs
from analysis.metadata import summary
from analysis.selector.filters.coverage_constraints import coverage_constraints
from analysis.selector.models.counter import Counter
from analysis.selector.models.filter import Filter
from analysis.selector.models.track import Track
from analysis.selector.models.window import Window
from analysis.utils import tile_group
from common.maths.tessellate import Tessellate

SETTINGS = configs.load()
GRID = Tessellate.of(SETTINGS.tile_km)


def trimmed(
    track: Track, window: Window, criteria: Filter
) -> tuple[list[int], list[int], list[int]] | None:
    """Drop the observations a tile does not need, keeping the most recent or the best.

    Args:
        track: The tile's observations on one time axis.
        window: The window the windowed instruments are counted inside.
        criteria: The filter read against the tile, its constraints and its bar.

    Returns:
        kept: The window's observations worth keeping, oldest first.
        standing: The timeless observations worth keeping, oldest first.
        reached: The cells each windowed constraint reaches once the rest are gone.
        Or None when the timeless looks with a distortion no longer meet their bar.
    """
    answering = {
        owner for owner, iid in enumerate(track.iids) if iid in criteria.timeless
    }
    # List of the observations that are kept
    kept = [
        index
        for index in range(window.first, window.last + 1)
        if track.owners[index] not in answering
    ]
    standing = sharad_drop_order(
        track, [index for index, owner in enumerate(track.owners) if owner in answering]
    )
    # Count what the window and the SHARAD looks hold in cells
    counter = Counter.empty(track.iids, track.grid.cells)
    for index in kept + standing:
        counter.hold(track.owners[index], track.cells[index])
    if coverage_constraints(criteria.standing, counter.cells_reached) is None:
        return None
    constraints = criteria.windowed + criteria.standing
    # Try to drop each observation, oldest first, SHARAD worst first, keep the rest
    dropped: set[int] = set()
    for index in kept + standing:
        owner, cells = track.owners[index], track.cells[index]
        filled = counter.observations_per_cell[owner]
        alone = int(np.count_nonzero(filled[cells] == 1))
        if alone >= criteria.gain(track.iids[owner], cells.size):
            continue
        counter.release(owner, cells)
        # If the window can do without the observation
        if coverage_constraints(constraints, counter.cells_reached) is not None:
            dropped.add(index)
        else:
            counter.hold(owner, cells)
    reached = coverage_constraints(criteria.windowed, counter.cells_reached)
    return (
        [index for index in kept if index not in dropped],
        sorted(index for index in standing if index not in dropped),
        reached,
    )


def sharad_drop_order(track: Track, looks: list[int]) -> list[int]:
    """Order the SHARAD looks with a distortion over the tile worst first.

    Args:
        track: The tile's admissible observations on one time axis.
        looks: The SHARAD looks to order, as indices into the track.

    Returns:
        ordered: The looks with a distortion, day only before night, then the most
            distorted, then the fewest cells first.
    """
    name = track.observations[0].tile
    tile = GRID.tile_named(name)
    group = tile_group.group_name(len(GRID.columns), tile, SETTINGS.tile_group_deg)
    distortions = {
        one.pdsid: one for one in summary.read_distortions(group) if one.tile == name
    }

    def rank(index: int) -> tuple[bool, float, int]:
        """Return how bad one look is, the higher the sooner it is dropped."""
        held = distortions[track.observations[index].pdsid]
        day = held.night is None
        return (day, held.overall if day else held.night, -track.cells[index].size)

    rated = [index for index in looks if track.observations[index].pdsid in distortions]
    return sorted(rated, key=rank, reverse=True)
