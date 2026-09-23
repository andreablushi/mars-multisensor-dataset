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
) -> tuple[list[int], list[int], list[int]]:
    """Drop the observations a tile does not need, keeping the most recent or the best.

    Args:
        track: The tile's observations on one time axis.
        window: The window the windowed instruments are counted inside.
        criteria: The filter read against the tile, its constraints and its bar.

    Returns:
        kept: The window's observations worth keeping, oldest first.
        standing: The timeless observations worth keeping, whenever they came.
        reached: The cells each windowed constraint reaches once the rest are gone.
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
    standing = [index for index, owner in enumerate(track.owners) if owner in answering]
    # Count what the window and the SHARAD looks hold in cells
    counter = Counter.empty(track.iids, track.grid.cells)
    for index in kept + standing:
        counter.hold(track.owners[index], track.cells[index])
    constraints = criteria.windowed + criteria.standing
    # Try to drop each observation, oldest first, SHARAD worst first, keep the rest
    dropped: set[int] = set()
    for index in kept + sharad_drop_order(track, standing):
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
        [index for index in standing if index not in dropped],
        reached,
    )


def sharad_drop_order(track: Track, looks: list[int]) -> list[int]:
    """Order SHARAD looks worst first, so the best over the same ground is kept.

    Args:
        track: The tile's admissible observations on one time axis.
        looks: The SHARAD looks to order, as indices into the track.

    Returns:
        ordered: The day side before the night, the most distorted and oldest first.
    """
    tile = GRID.tile_named(track.observations[0].tile)
    group = tile_group.group_name(len(GRID.columns), tile, SETTINGS.tile_group_deg)
    distortions = {one.pdsid: one for one in summary.read_distortions(group)}

    def rank(index: int) -> tuple[int, float, int]:
        """Return where one look stands, the night side before the rest."""
        held = distortions.get(track.observations[index].pdsid)
        if held is None or held.overall is None:
            return (2, 0.0, -index)
        if held.night is None:
            return (1, held.overall, -index)
        return (0, held.night, -index)

    return sorted(looks, key=rank, reverse=True)
