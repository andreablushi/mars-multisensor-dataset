"""Which observations a tile can do without, and what the rest still reach."""

from __future__ import annotations

import numpy as np

from analysis.metadata.loaders.ancillary import read_distortions
from analysis.models.ancillary import Distortion
from analysis.selector.filters.coverage_constraints import cells_per_constraint
from analysis.selector.models.counter import Counter
from analysis.selector.models.criteria import Criteria
from analysis.selector.models.track import Track
from analysis.utils.tile_group import group_of_tile_named

HYPERSPECTRAL = "hsp"


def trimmed_window(
    track: Track, first: int, last: int, criteria: Criteria
) -> tuple[list[int], list[int], list[int]]:
    """Drop the observations a tile does not need, keeping the most recent or the best.

    Args:
        track: The tile's observations on one time axis.
        first: The index of the earliest observation the window holds.
        last: The index of the latest one.
        criteria: The criteria, holding the timeless instruments and redundant shares.

    Returns:
        kept: The window's observations worth keeping, oldest first.
        standing: The timeless observations worth keeping, oldest first.
        reached: The cells each windowed constraint reaches once the rest are gone.
    """
    timeless_owners = {
        owner for owner, iid in enumerate(track.iids) if iid in criteria.timeless
    }
    # List of the observations that are kept
    kept = [
        index
        for index in range(first, last + 1)
        if track.owners[index] not in timeless_owners
    ]
    standing = sharad_dropped_first(
        track,
        [index for index, owner in enumerate(track.owners) if owner in timeless_owners],
    )
    # Count what the window and the SHARAD looks hold in cells
    counter = Counter.over(track, kept + standing)
    constraints = track.windowed + track.standing
    # A hyperspectral look goes last, so it outlasts a multispectral one
    windowed_order = sorted(
        kept,
        key=lambda index: (
            track.observations[index].pdsid.startswith(HYPERSPECTRAL),
            index,
        ),
    )
    # Keep the best first, then drop each later one matching a look already kept
    dropped: set[int] = set()
    retained: list[int] = []
    for index in reversed(windowed_order + standing):
        owner, cells = track.owners[index], track.cells[index]
        share = criteria.redundant_share_threshold.get(track.iids[owner], 1.0)
        filled = np.zeros(track.grid.cell_count, dtype=bool)
        filled[cells] = True
        matched = any(
            np.count_nonzero(filled[track.cells[other]])
            > share * min(cells.size, track.cells[other].size)
            for other in retained
            if track.owners[other] == owner
        )
        if matched:
            counter.release(owner, cells)
            # If the window can do without the observation
            if cells_per_constraint(constraints, counter.cells_reached) is not None:
                dropped.add(index)
                continue
            counter.hold(owner, cells)
        retained.append(index)
    reached = cells_per_constraint(track.windowed, counter.cells_reached)
    return (
        [index for index in kept if index not in dropped],
        sorted(index for index in standing if index not in dropped),
        reached,
    )


def sharad_dropped_first(track: Track, indices: list[int]) -> list[int]:
    """Order the SHARAD looks worst first, so the best over the same ground is kept.

    Args:
        track: The tile's admissible observations on one time axis.
        indices: The SHARAD looks to order, as indices into the track.

    Returns:
        ordered: The looks without a distortion over the tile first, oldest first,
            then day only before night, the most distorted and fewest cells first.
    """
    tile = track.observations[0].tile
    distortions = {
        distortion.pdsid: distortion
        for distortion in read_distortions(group_of_tile_named(tile))
        if distortion.tile == tile
    }
    return sorted(
        indices,
        key=lambda index: sharad_rank(track, distortions, index),
        reverse=True,
    )


def sharad_rank(
    track: Track, distortions: dict[str, Distortion], index: int
) -> tuple[int, float, int]:
    """Rank how bad one SHARAD look is.

    Args:
        track: The tile's admissible observations on one time axis.
        distortions: The distortion of each look over the tile, by pdsid.
        index: The look to rank, as its index into the track.

    Returns:
        rank: How bad it is, the higher the sooner it is dropped.
    """
    distortion = distortions.get(track.observations[index].pdsid)
    if distortion is None:
        return (2, 0.0, -index)
    if distortion.night is None:
        return (1, distortion.overall, -track.cells[index].size)
    return (0, distortion.night, -track.cells[index].size)
