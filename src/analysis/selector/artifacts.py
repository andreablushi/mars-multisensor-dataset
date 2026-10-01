"""The written selection: which tiles earned a place, and what to download."""

from __future__ import annotations

from collections.abc import Sequence

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
