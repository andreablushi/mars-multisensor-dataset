"""What one companion table says over each group its product was listed in."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from analysis.ground_truth import box
from analysis.models.companion import Companion
from analysis.models.tile_group import TileGroup
from building.common.pds import tables

Summary = dict[str, float]

Ledger = dict[str, dict[str, Summary]]


def summarise_table(
    table: Path,
    columns: list[dict[str, str]],
    row_bytes: int,
    companion: Companion,
    groups: Mapping[str, TileGroup],
) -> dict[str, Summary]:
    """Read the median of every asked column over the rows inside each group.

    Args:
        table: The fixed width table, one row per sample of its product.
        columns: The COLUMN objects of the placing and asked columns alone.
        row_bytes: How many bytes each row takes, its line ending counted.
        companion: What the companion is, and which of its columns are asked for.
        groups: The groups the product still has to be summarised over, by name.

    Returns:
        summaries: Each group's medians by field, empty where no row fell in it.
    """
    rows = table.stat().st_size // row_bytes
    read = tables.build_table(
        table, {"ROWS": str(rows), "ROW_BYTES": str(row_bytes)}, columns
    )
    latitude = read[companion.latitude]
    # A row is a point, so it is tested as a box of no size
    point = (latitude, latitude, read[companion.longitude] % 360.0, 0.0)
    summaries: dict[str, Summary] = {}
    for name, group in groups.items():
        inside = box.inside(point, box.bounds_box(group))
        summaries[name] = (
            {
                field: round(float(np.median(read[column][inside])), 3)
                for field, column in companion.columns.items()
            }
            if inside.any()
            else {}
        )
    return summaries
