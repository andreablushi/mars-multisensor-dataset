"""Reading the written selection back, which is how the building half reads it."""

from __future__ import annotations

import pyarrow.parquet as pq

from analysis import paths
from analysis.selector.artifacts.write import OBSERVATIONS, TILES
from analysis.selector.models.selection import (
    SelectedObservation,
    SelectedTile,
    Selection,
)


def read_selected_tiles() -> list[SelectedTile]:
    """Read back every tile the selection stage searched, without what each keeps.

    Returns:
        tiles: One row per tile searched, in the order they were written.

    Raises:
        FileNotFoundError: When no selection has been written there.
    """
    tiles = paths.SELECTED_TILES_PATH
    if not tiles.is_file():
        raise FileNotFoundError(f"no selection was written in {tiles.parent}")
    return [
        SelectedTile(**row) for row in pq.read_table(tiles, schema=TILES).to_pylist()
    ]


def read_dataset_list() -> list[Selection]:
    """Read back every tile the selection stage searched, and what each keeps.

    Returns:
        selections: One entry per tile searched, in the order they were written.

    Raises:
        FileNotFoundError: When no selection has been written there.
    """
    observations = paths.SELECTED_OBSERVATIONS_PATH
    if not observations.is_file():
        raise FileNotFoundError(f"no selection was written in {observations.parent}")
    kept: dict[str, list[SelectedObservation]] = {}
    for row in pq.read_table(observations, schema=OBSERVATIONS).to_pylist():
        observation = SelectedObservation(**row)
        kept.setdefault(observation.tile, []).append(observation)
    return [
        Selection(tile=one, observations=kept.get(one.tile, []))
        for one in read_selected_tiles()
    ]
