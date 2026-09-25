"""One tile as the selection left it, and what the observations it keeps left on it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from analysis.coverage.models.coverage import SetCoverage
from analysis.selector.filters.admit import landed_pixels
from analysis.selector.merge import merge_track
from analysis.selector.models.selection import Selection
from analysis.stats.artifacts import selection_by_tile
from analysis.stats.models import InstrumentReach, TileStats, TileTrack
from common.config import analysis_settings

# How many tiles are held read at once, so every panel of one shares it.
TILE_CACHE = 8

# The tiles held read, so every panel of one shares the reading
_tracks_read: dict[str, TileTrack | None] = {}


def read_tile_track(coverage: Sequence[SetCoverage]) -> TileTrack | None:
    """Read one tile as the selection left it, however many panels ask for it.

    Args:
        coverage: The tile's instrument sets, in the order they are drawn.

    Returns:
        tile_track: Its track and kept observations, or None if nothing is measurable.

    Raises:
        FileNotFoundError: When no selection has been written to read it off.
    """
    key = coverage[0].summary.tile
    if key not in _tracks_read:
        if len(_tracks_read) >= TILE_CACHE:
            _tracks_read.clear()
        selection = selection_by_tile().get(key)
        _tracks_read[key] = (
            None if selection is None else track_tile(coverage, selection)
        )
    return _tracks_read[key]


def track_tile(
    coverage: Sequence[SetCoverage], selection: Selection
) -> TileTrack | None:
    """Place the observations one tile keeps on the track they were taken over.

    Args:
        coverage: The tile's instrument sets, in any order.
        selection: What the selection left of it, and the observations it keeps.

    Returns:
        tile_track: Its track and where its kept observations sit, or None if
            nothing is measurable.
    """
    track = merge_track(coverage, analysis_settings().window)
    if track is None:
        return None
    index_of = {
        observation.pdsid: index for index, observation in enumerate(track.observations)
    }
    return TileTrack(
        track=track,
        window=selection.tile,
        taken=tuple(
            sorted(
                index_of[observation.pdsid]
                for observation in selection.observations
                if observation.pdsid in index_of
            )
        ),
    )


def measure_tile(tile_track: TileTrack) -> TileStats:
    """Measure what the instruments left on one tile, given the observations it keeps.

    Args:
        tile_track: Its track, its window, and where its kept observations sit.

    Returns:
        stats: What it holds.
    """
    track = tile_track.track
    # What each instrument left inside the window, and which of them each cell holds
    cells_by_iid: dict[str, set[int]] = {}
    observations_by_iid: dict[str, int] = {}
    iids_by_cell: dict[int, set[str]] = {}
    pixels_by_iid: dict[str, float | None] = {}
    for index in tile_track.taken:
        iid = track.iids[track.owners[index]]
        cells_by_iid.setdefault(iid, set()).update(track.cells[index])
        observations_by_iid[iid] = observations_by_iid.get(iid, 0) + 1
        for cell in track.cells[index].tolist():
            iids_by_cell.setdefault(cell, set()).add(iid)
        observation = track.observations[index]
        landed = pixels_by_iid.get(iid, 0.0)
        if landed is None or observation.pixels is None or not observation.own_km2:
            pixels_by_iid[iid] = None
        else:
            pixels_by_iid[iid] = landed + landed_pixels(
                observation, len(track.cells[index]), track.grid.cell_km2
            )
    overlaps: dict[tuple[str, ...], float] = {}
    for cell in sorted(iids_by_cell):
        instrument_names = tuple(sorted(iids_by_cell[cell]))
        overlaps[instrument_names] = (
            overlaps.get(instrument_names, 0.0) + track.grid.cell_km2
        )
    # A pixel is one size whether or not its look was chosen, so all are read
    pixel_km2: dict[str, float] = {}
    for index, owner in enumerate(track.owners):
        iid = track.iids[owner]
        observation = track.observations[index]
        if iid not in pixel_km2 and observation.pixels and observation.own_km2:
            pixel_km2[iid] = observation.own_km2 / observation.pixels
    return TileStats(
        window=tile_track.window,
        iids=list(dict.fromkeys(track.iids)),
        pixel_km2=pixel_km2,
        reached={
            iid: InstrumentReach(
                km2=len(cells_reached) * track.grid.cell_km2,
                pixels=pixels_by_iid[iid],
                observations_taken=observations_by_iid[iid],
            )
            for iid, cells_reached in cells_by_iid.items()
        },
        overlaps=dict(sorted(overlaps.items(), key=lambda ground: -ground[1])),
    )


def ground_by_instrument_count(
    overlaps: Mapping[tuple[str, ...], float],
) -> dict[int, float]:
    """Add up the ground each number of instruments reaches at once.

    Args:
        overlaps: The ground each set of instruments reaches, counting a cell once.

    Returns:
        ground: The ground in km2 by how many instruments reach it, fewest first.
    """
    summed: dict[int, float] = {}
    for instrument_names, km2 in overlaps.items():
        summed[len(instrument_names)] = summed.get(len(instrument_names), 0.0) + km2
    return dict(sorted(summed.items()))
