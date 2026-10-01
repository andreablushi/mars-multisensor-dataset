"""The dataset selection: every measured tile searched, its rows, and its stats."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from analysis import console
from analysis.coverage import artifacts as coverage_artifacts
from analysis.selector.artifacts import write_selection
from analysis.selector.merge import merge_track
from analysis.selector.models.selection import (
    SelectedObservation,
    SelectedTile,
    Selection,
)
from analysis.selector.models.survey import Survey
from analysis.selector.models.track import Track
from analysis.selector.search import best_survey
from analysis.stats.models import TileStats, TileTrack
from analysis.stats.tile import measure_tile
from analysis.utils.tile_group import tile_grid
from common.config import analysis_settings
from common.maths import box
from common.models.tile import Tile
from common.pool import cancellable_pool


def select_dataset(workers: int) -> tuple[list[Selection], list[TileStats]]:
    """Search every measured tile under the criteria, and write the selection out.

    Args:
        workers: How many processes to search on at once, as the run is configured.

    Returns:
        selections: What the search left of each tile, band by band, west to east.
        measured: What the observations each tile keeps left on it, a group at a
            time, for every tile with something to measure.
    """
    groups = coverage_artifacts.measured_groups()
    selections: list[Selection] = []
    measured: list[TileStats] = []
    with cancellable_pool(ProcessPoolExecutor(workers)) as pool:
        searched = pool.map(_group_selections_and_stats, groups, chunksize=1)
        for done, (group_selections, group_stats) in enumerate(searched, 1):
            selections.extend(group_selections)
            measured.extend(group_stats)
            console.print_progress("selection", done, len(groups))
    selections.sort(key=lambda selection: (selection.tile.band, selection.tile.column))
    write_selection(selections)
    return selections, measured


def tile_selection(survey: Survey | None, track: Track | None, tile: Tile) -> Selection:
    """Read one tile's search as the rows the selection is written from.

    Args:
        survey: The window the tile earned, or None where it earned none.
        track: Its admissible observations on one time axis, or None.
        tile: The tile itself, its box carried so later runs need only the selection.

    Returns:
        selection: Its own row, and a row for each observation it keeps.
    """
    row = SelectedTile(
        tile=tile.name,
        band=tile.band,
        column=tile.column,
        **box.box_edges(tile),
        kept=survey is not None,
        area_km2=track.grid.area_km2 if track else 0.0,
        start=survey.start if survey else None,
        end=survey.end if survey else None,
        days=survey.days if survey else 0.0,
        geo_mean=survey.geo_mean if survey else 0.0,
        taken=len(survey.taken) if survey else 0,
    )
    if survey is None:
        return Selection(tile=row)
    observations = []
    for index in survey.taken:
        observation = track.observations[index]
        observations.append(
            SelectedObservation(
                tile=tile.name,
                ihid=observation.ihid,
                iid=observation.iid,
                pt=observation.pt,
                pdsid=observation.pdsid,
                t_start=observation.t_start,
                standing=index in survey.standing,
            )
        )
    return Selection(tile=row, observations=observations)


def _group_selections_and_stats(
    group: str,
) -> tuple[list[Selection], list[TileStats]]:
    """Search every tile one group measured, and measure what each keeps.

    Args:
        group: The name of the tile group.

    Returns:
        selections: The rows of every tile a measured set reached, in the order read.
        measured: What the observations each tile keeps left on it, for every tile
            with something to measure.
    """
    criteria = analysis_settings().criteria
    grid = tile_grid()
    selections, measured = [], []
    for name, coverage in coverage_artifacts.read_group_coverage(group).items():
        track = merge_track(coverage, criteria)
        survey = best_survey(track, criteria) if track else None
        selection = tile_selection(survey, track, grid.tile_named(name))
        selections.append(selection)
        if track:
            taken = survey.taken if survey else ()
            measured.append(measure_tile(TileTrack(track, selection.tile, taken)))
    return selections, measured
