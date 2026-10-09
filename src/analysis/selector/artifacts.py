"""The written selection: which tiles earned a place, and what to download."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

from analysis import paths
from analysis.selector.models.selection import (
    SelectedObservation,
    SelectedTile,
    Selection,
)
from common.disk import parquet

TILES = parquet.schema_of(SelectedTile)
OBSERVATIONS = parquet.schema_of(SelectedObservation)


def write_selection(selections: Sequence[Selection]) -> None:
    """Write every searched tile down, and every observation they keep.

    Args:
        selections: What the search left of each tile, in the order to write them.
    """
    parquet.write_rows(
        [selection.tile for selection in selections], TILES, paths.SELECTED_TILES_PATH
    )
    parquet.write_rows(
        [
            observation
            for selection in selections
            for observation in selection.observations
        ],
        OBSERVATIONS,
        paths.SELECTED_OBSERVATIONS_PATH,
    )


def read_selected_tiles() -> list[SelectedTile]:
    """Read back every tile the selection stage searched, without what each keeps."""
    return parquet.read_rows(SelectedTile, TILES, paths.SELECTED_TILES_PATH)


def read_selection() -> list[Selection]:
    """Read back every tile the selection stage searched, and what each keeps.

    Returns:
        selections: One entry per tile searched, in the order they were written.

    Raises:
        FileNotFoundError: When no selection has been written there.
    """
    observations_by_tile: dict[str, list[SelectedObservation]] = {}
    for observation in parquet.read_rows(
        SelectedObservation, OBSERVATIONS, paths.SELECTED_OBSERVATIONS_PATH
    ):
        observations_by_tile.setdefault(observation.tile, []).append(observation)
    return [
        Selection(tile=tile, observations=observations_by_tile.get(tile.tile, []))
        for tile in read_selected_tiles()
    ]


def read_refused_observations() -> list[SelectedObservation]:
    """Read back every kept observation a build cropped empty, none before a build."""
    if not paths.REFUSED_OBSERVATIONS_PATH.exists():
        return []
    return parquet.read_rows(
        SelectedObservation, OBSERVATIONS, paths.REFUSED_OBSERVATIONS_PATH
    )


def write_refused_observations(refused: Sequence[SelectedObservation]) -> None:
    """Add observations a build cropped empty to those refused before, once each."""
    parquet.write_rows(
        list(dict.fromkeys([*read_refused_observations(), *refused])),
        OBSERVATIONS,
        paths.REFUSED_OBSERVATIONS_PATH,
    )


@cache
def refused_pdsids() -> dict[str, frozenset[str]]:
    """Return the products a build cropped empty, by the tile they were kept for."""
    by_tile: dict[str, set[str]] = {}
    for observation in read_refused_observations():
        by_tile.setdefault(observation.tile, set()).add(observation.pdsid)
    return {tile: frozenset(pdsids) for tile, pdsids in by_tile.items()}
