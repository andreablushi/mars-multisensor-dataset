"""What every observation offered to a tile landed on it, and the bar it faced."""

from __future__ import annotations

from analysis.selector.filters.admissible import landed_pixels
from analysis.stats.models.landing import Landed
from analysis.stats.models.tile import TileLooks


def landed_per_set(looks: TileLooks) -> list[Landed]:
    """Read what every observation offered to one tile landed on it.

    Args:
        looks: Its timeline and the filter it was read under.

    Returns:
        landed: One entry per instrument set, in the order the track indexes them.
    """
    track = looks.track
    counted: list[list[float]] = [[] for _ in track.labels]
    for index, owner in enumerate(track.owners):
        counted[owner].append(
            landed_pixels(
                track.observations[index],
                len(track.cells[index]),
                track.grid.cell_km2,
            )
        )
    for observation, owner, cells in track.refused:
        counted[owner].append(
            landed_pixels(observation, len(cells), track.grid.cell_km2)
        )
    return [
        Landed(
            label=track.labels[owner],
            iid=track.iids[owner],
            counts=sorted(counted[owner]),
            bar=track.least[owner],
        )
        for owner in range(len(track.labels))
    ]
