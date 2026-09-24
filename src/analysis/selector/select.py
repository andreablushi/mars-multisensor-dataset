"""Selecting the dataset: every measured tile searched, and what is kept."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from analysis import console
from analysis.coverage.artifacts import index
from analysis.selector.artifacts import write
from analysis.selector.models.selection import (
    SelectedObservation,
    SelectedTile,
    Selection,
)
from analysis.selector.models.survey import Study
from analysis.utils.tile_group import tile_grid
from common.config import analysis_settings
from common.models.tile import Tile


def select_dataset(workers: int) -> list[Selection]:
    """Search every measured tile under the filter, and write the selection out.

    Args:
        workers: How many processes to search on at once, as the run is configured.

    Returns:
        picked: What the search left of each tile, band by band, west to east.
    """
    groups = index.measured_groups()
    picked: list[Selection] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        searched = pool.map(_searched, groups, chunksize=1)
        for done, found in enumerate(searched, 1):
            picked.extend(found)
            console.report("selection", done, len(groups))
    picked.sort(key=lambda one: (one.tile.band, one.tile.column))
    write.write_selection(picked)
    return picked


def selected(study: Study, tile: Tile) -> Selection:
    """Read one tile's search as the rows the selection is written from.

    Args:
        study: What the search found over it.
        tile: The tile itself, its box carried so later runs need only the selection.

    Returns:
        selection: Its own row, and a row for each observation it keeps.
    """
    survey, track = study.survey, study.track
    row = SelectedTile(
        tile=tile.name,
        band=tile.band,
        column=tile.column,
        min_lat=tile.min_lat,
        max_lat=tile.max_lat,
        west_lon=tile.west_lon,
        east_lon=tile.east_lon,
        kept=survey is not None,
        area_km2=track.grid.area_km2 if track else 0.0,
        start=survey.start if survey else None,
        end=survey.end if survey else None,
        days=survey.days if survey else 0.0,
        geo_mean=survey.geo_mean if survey else 0.0,
        taken=len(survey.taken) if survey else 0,
    )
    if survey is None or track is None:
        return Selection(tile=row)
    standing = set(survey.standing)
    return Selection(
        tile=row,
        observations=[
            SelectedObservation(
                tile=tile.name,
                ihid=track.observations[at].ihid,
                iid=track.observations[at].iid,
                pt=track.observations[at].pt,
                pdsid=track.observations[at].pdsid,
                t_start=track.observations[at].t_start,
                standing=at in standing,
            )
            for at in survey.taken
        ],
    )


def _searched(group: str) -> list[Selection]:
    """Search every tile one group measured and read each as the rows it is written as.

    Args:
        group: The name of the tile group.

    Returns:
        selections: The rows of every tile a measured set reached, in the order read.
    """
    window = analysis_settings().window
    return [
        selected(Study.over(coverage, window), tile_grid().tile_named(name))
        for name, coverage in index.load_group(group).items()
    ]
