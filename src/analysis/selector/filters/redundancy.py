"""Which observations a tile can do without, and what the rest still reach."""

from __future__ import annotations

import numpy as np

from analysis.metadata import summary
from analysis.models.ancillary import Distortion
from analysis.selector.filters.coverage_constraints import cells_per_constraint
from analysis.selector.models.counter import Counter
from analysis.selector.models.filter import Filter
from analysis.selector.models.track import Track
from analysis.utils.tile_group import group_of, tile_grid

HYPERSPECTRAL = "hsp"


def trimmed(
    track: Track, first: int, last: int, criteria: Filter
) -> tuple[list[int], list[int], list[int]]:
    """Drop the observations a tile does not need, keeping the most recent or the best.

    Args:
        track: The tile's observations on one time axis.
        first: The index of the earliest observation the window holds.
        last: The index of the latest one.
        criteria: The filter, holding the timeless instruments and redundant shares.

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
    standing = sharad_drop_order(
        track,
        [index for index, owner in enumerate(track.owners) if owner in timeless_owners],
    )
    # Count what the window and the SHARAD looks hold in cells
    counter = Counter.over(track, kept + standing)
    constraints = track.windowed + track.standing
    # Drop each redundant observation but the best of its group, oldest or worst first
    dropped: set[int] = set()
    for indices in (kept, standing):
        for group in redundant_groups(track, indices, criteria):
            for index in group[:-1]:
                owner, cells = track.owners[index], track.cells[index]
                counter.release(owner, cells)
                # If the window can do without the observation
                if cells_per_constraint(constraints, counter.cells_reached) is not None:
                    dropped.add(index)
                else:
                    counter.hold(owner, cells)
    reached = cells_per_constraint(track.windowed, counter.cells_reached)
    return (
        [index for index in kept if index not in dropped],
        sorted(index for index in standing if index not in dropped),
        reached,
    )


def redundant_groups(
    track: Track, indices: list[int], criteria: Filter
) -> list[list[int]]:
    """Gather the looks redundant with each other, directly or through another look.

    Args:
        track: The tile's observations on one time axis.
        indices: The looks to gather, as indices into the track, worst first.
        criteria: The filter holding the share past which two looks are redundant.

    Returns:
        groups: Each group of one instrument's redundant looks, worst first and
            every hyperspectral look after every multispectral one.
    """
    filled = np.zeros((len(indices), track.grid.cells), dtype=np.float32)
    for row, index in enumerate(indices):
        filled[row, track.cells[index]] = 1.0
    shared = filled @ filled.T
    sizes = np.diag(shared)
    owners = np.array([track.owners[index] for index in indices])
    thresholds = np.array(
        [
            criteria.redundant_share_threshold.get(track.iids[owner], 1.0)
            for owner in owners
        ]
    )
    overlapping = shared > (thresholds * sizes)[:, None]
    linked = (overlapping | overlapping.T) & (owners[:, None] == owners[None, :])
    groups: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(indices)):
        if start in seen:
            continue
        members, frontier = {start}, [start]
        while frontier:
            fresh = set(np.flatnonzero(linked[frontier].any(axis=0)).tolist()) - members
            members |= fresh
            frontier = list(fresh)
        seen |= members
        # A hyperspectral look goes last, so it outlasts a multispectral one
        ordered = sorted(
            members,
            key=lambda row: (
                track.observations[indices[row]].pdsid.startswith(HYPERSPECTRAL),
                row,
            ),
        )
        groups.append([indices[row] for row in ordered])
    return groups


def sharad_drop_order(track: Track, indices: list[int]) -> list[int]:
    """Order the SHARAD looks worst first, so the best over the same ground is kept.

    Args:
        track: The tile's admissible observations on one time axis.
        indices: The SHARAD looks to order, as indices into the track.

    Returns:
        ordered: The looks without a distortion over the tile first, oldest first,
            then day only before night, the most distorted and fewest cells first.
    """
    tile = track.observations[0].tile
    group = group_of(tile_grid().tile_named(tile))
    distortions = {
        distortion.pdsid: distortion
        for distortion in summary.read_distortions(group)
        if distortion.tile == tile
    }
    return sorted(
        indices,
        key=lambda index: sharad_drop_rank(track, distortions, index),
        reverse=True,
    )


def sharad_drop_rank(
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
