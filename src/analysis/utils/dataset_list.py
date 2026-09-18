"""Reading the written selection back, which is how the building half reads it."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from analysis import paths
from analysis.selector.artifacts.write import OBSERVATIONS, TILES
from analysis.selector.models.selection import (
    SelectedObservation,
    SelectedTile,
    Selection,
)


def read_dataset_list(root: Path = paths.SELECTION_ROOT) -> list[Selection]:
    """Read back every tile the selection stage searched, and what each keeps.

    Args:
        root: The directory the selection was written in.

    Returns:
        selections: One entry per tile searched, in the order they were written.

    Raises:
        FileNotFoundError: When no selection has been written there.
    """
    tiles = root / paths.SELECTED_TILES_NAME
    observations = root / paths.SELECTED_OBSERVATIONS_NAME
    if not tiles.is_file() or not observations.is_file():
        raise FileNotFoundError(f"no selection was written in {root}")
    kept: dict[str, list[SelectedObservation]] = {}
    for row in pq.read_table(observations, schema=OBSERVATIONS).to_pylist():
        observation = SelectedObservation(**row)
        kept.setdefault(observation.tile, []).append(observation)
    return [
        Selection(tile=SelectedTile(**row), observations=kept.get(row["tile"], []))
        for row in pq.read_table(tiles, schema=TILES).to_pylist()
    ]
