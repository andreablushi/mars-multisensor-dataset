"""What each instrument set brought one tile, observation by observation, over time."""

from __future__ import annotations

from collections.abc import Sequence

from analysis.coverage.models.coverage import SetCoverage
from analysis.selector.filters.admit import landed_pixels
from analysis.stats.models import Landing, TileTrack, Timeline


def landings_per_set(tile_track: TileTrack) -> list[Landing]:
    """Read what every observation offered to one tile landed on it.

    Args:
        tile_track: Its track, which holds the observations turned away too.

    Returns:
        landings: One entry per instrument set, in the order the track indexes them.
    """
    track = tile_track.track
    counted: list[list[float]] = [[] for _ in track.labels]
    offered = [*zip(track.observations, track.owners, track.cells), *track.refused]
    for observation, owner, cells in offered:
        counted[owner].append(
            landed_pixels(observation, len(cells), track.grid.cell_km2)
        )
    return [
        Landing(
            label=track.labels[owner],
            iid=track.iids[owner],
            counts=sorted(counted[owner]),
            bar=track.min_pixels[owner],
        )
        for owner in range(len(track.labels))
    ]


def timelines_per_set(coverage: Sequence[SetCoverage]) -> list[Timeline]:
    """Read every instrument set's observations of the whole tile.

    Args:
        coverage: The tile's instrument sets, in the order they are drawn.

    Returns:
        timelines: One timeline per set, in the same order.
    """
    area_km2 = coverage[0].summary.tile_area_km2
    first = min(instrument.summary.t_first for instrument in coverage)
    last = max(instrument.summary.t_last for instrument in coverage)
    return [
        Timeline(
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
