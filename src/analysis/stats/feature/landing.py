"""What every observation offered to a feature landed on it, and the bar it faced."""

from __future__ import annotations

from analysis.coverage.models.coverage import Event
from analysis.stats.models.feature import FeatureLooks
from analysis.stats.models.landing import Landed


def landed_per_set(looks: FeatureLooks) -> list[Landed]:
    """Read what every observation offered to one feature landed on it.

    Args:
        looks: Its timeline and the filter it was read under.

    Returns:
        landed: One entry per instrument set, in the order the track indexes them.
    """
    track = looks.track
    least = looks.criteria.least
    counted: list[list[float]] = [[] for _ in track.labels]
    for index, owner in enumerate(track.owners):
        counted[owner].append(
            _pixels(
                track.observations[index],
                len(track.cells[index]),
                track.grid.cell_km2,
            )
        )
    for observation, owner, cells in track.refused:
        counted[owner].append(_pixels(observation, len(cells), track.grid.cell_km2))
    return [
        Landed(
            label=track.labels[owner],
            iid=track.iids[owner],
            counts=sorted(counted[owner]),
            bar=least[owner],
        )
        for owner in range(len(track.labels))
    ]


def _pixels(observation: Event, cells: int, cell_km2: float) -> float:
    """Read how many pixels one observation landed inside the feature.

    Args:
        observation: The observation, carrying what it covered and what it landed.
        cells: How many of the feature's own cells its footprint fills.
        cell_km2: How much ground one of those cells covers.

    Returns:
        pixels: Its pixels, scaled to the part of its footprint the feature holds.
    """
    if not observation.own_km2:
        return 0.0
    return observation.pixels * cells * cell_km2 / observation.own_km2
