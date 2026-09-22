"""Writing the selection down: which tiles earned a place, and what to download."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from analysis import paths
from analysis.selector.models.selection import (
    SelectedObservation,
    SelectedTile,
    Selection,
)
from common.disk import parquet

TILES = parquet.schema_of(SelectedTile)
OBSERVATIONS = parquet.schema_of(SelectedObservation)


def write_selection(
    picked: Sequence[Selection], root: Path = paths.SELECTION_ROOT
) -> tuple[Path, Path]:
    """Write every searched tile down, and every observation they keep.

    Args:
        picked: What the search left of each tile, in the order to write them.
        root: The directory the two files are written in, made when it is missing.

    Returns:
        tiles: The file the tiles were written to.
        observations: The file the observations were written to.
    """
    tiles = root / paths.SELECTED_TILES_NAME
    observations = root / paths.SELECTED_OBSERVATIONS_NAME
    parquet.write([one.tile for one in picked], TILES, tiles)
    parquet.write(
        [kept for one in picked for kept in one.observations],
        OBSERVATIONS,
        observations,
    )
    return tiles, observations
