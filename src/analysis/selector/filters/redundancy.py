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

HYPERSPECTRAL = "hsp"


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
        standing: The timeless observations worth keeping, oldest first.
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
    standing = sharad_drop_order(
        track, [index for index, owner in enumerate(track.owners) if owner in answering]
    )
    # Count what the window and the SHARAD looks hold in cells
    counter = Counter.empty(track.iids, track.grid.cells)
    for index in kept + standing:
        counter.hold(track.owners[index], track.cells[index])
    constraints = criteria.windowed + criteria.standing
    # Drop each redundant observation but the best of its group, oldest or worst first
    dropped: set[int] = set()
    for looks in (kept, standing):
        for group in redundant_groups(track, looks, criteria):
            for index in group[:-1]:
                owner, cells = track.owners[index], track.cells[index]
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


def redundant_groups(
    track: Track, looks: list[int], criteria: Filter
) -> list[list[int]]:
    """Gather the looks redundant with each other, directly or through another look.

    Args:
        track: The tile's observations on one time axis.
        looks: The looks to gather, worst first.
        criteria: The filter holding the share past which two looks are redundant.

    Returns:
        groups: Each group of one instrument's redundant looks, worst first and
            every hyperspectral look after every multispectral one.
    """
    filled = np.zeros((len(looks), track.grid.cells), dtype=np.float32)
    for row, index in enumerate(looks):
        filled[row, track.cells[index]] = 1.0
    shared = filled @ filled.T
    sizes = np.diag(shared)
    owners = np.array([track.owners[index] for index in looks])
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
    for start in range(len(looks)):
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
                track.observations[looks[row]].pdsid.startswith(HYPERSPECTRAL),
                row,
            ),
        )
        groups.append([looks[row] for row in ordered])
    return groups


def sharad_drop_order(track: Track, looks: list[int]) -> list[int]:
    """Order the SHARAD looks worst first, so the best over the same ground is kept.

    Args:
        track: The tile's admissible observations on one time axis.
        looks: The SHARAD looks to order, as indices into the track.

    Returns:
        ordered: The looks without a distortion over the tile first, oldest first,
            then day only before night, the most distorted and fewest cells first.
    """
    settings = configs.load()
    grid = Tessellate.of(settings.tile_km)
    name = track.observations[0].tile
    tile = grid.tile_named(name)
    group = tile_group.group_name(len(grid.columns), tile, settings.tile_group_deg)
    distortions = {
        one.pdsid: one for one in summary.read_distortions(group) if one.tile == name
    }

    def rank(index: int) -> tuple[int, float, int]:
        """Return how bad one look is, the higher the sooner it is dropped."""
        held = distortions.get(track.observations[index].pdsid)
        if held is None:
            return (2, 0.0, -index)
        if held.night is None:
            return (1, held.overall, -track.cells[index].size)
        return (0, held.night, -track.cells[index].size)

    return sorted(looks, key=rank, reverse=True)
