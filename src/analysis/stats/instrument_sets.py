"""What each instrument set brought one tile, look by look and over time."""

from __future__ import annotations

from collections.abc import Sequence

from analysis.coverage.models.coverage import SetCoverage
from analysis.selector.filters.admissible import landed_pixels
from analysis.stats.models import Landed, Series, TileLooks


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


def coverage_over_time(coverage: Sequence[SetCoverage]) -> list[Series]:
    """Read every instrument set's observations of the whole tile.

    Args:
        coverage: The tile's instrument sets, in the order they are drawn.

    Returns:
        series: One series per set, in the same order.
    """
    area_km2 = coverage[0].summary.tile_area_km2
    first = min(instrument.summary.t_first for instrument in coverage)
    last = max(instrument.summary.t_last for instrument in coverage)
    return [
        Series(
            label=instrument.label,
            iid=instrument.summary.iid,
            times=[observation.t_start for observation in instrument.events],
            shares=[
                observation.own_km2 / area_km2 for observation in instrument.events
            ],
            running=[observation.cum_frac for observation in instrument.events],
            covered=instrument.summary.covered_frac,
            first=first,
            last=last,
            reason=instrument.reason,
        )
        for instrument in coverage
    ]
