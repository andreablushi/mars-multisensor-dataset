"""Writing the selection down: which tiles earned a place, and what to download."""

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


def write_selection(picked: Sequence[Selection]) -> None:
    """Write every searched tile down, and every observation they keep.

    Args:
        picked: What the search left of each tile, in the order to write them.
    """
    parquet.write([one.tile for one in picked], TILES, paths.SELECTED_TILES_PATH)
    parquet.write(
        [kept for one in picked for kept in one.observations],
        OBSERVATIONS,
        paths.SELECTED_OBSERVATIONS_PATH,
    )
