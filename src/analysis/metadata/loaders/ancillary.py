"""Loading the signal phase distortion one ancillary table holds over each tile."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from analysis.models.ancillary import Ancillary, Distortion
from analysis.models.tile_group import TileGroup
from building.common.pds import tables
from common.maths.tessellate import Tessellate


def load_distortions(
    table: Path,
    pdsid: str,
    label: dict[str, str],
    columns: list[dict[str, str]],
    ancillary: Ancillary,
    groups: Mapping[str, TileGroup],
    grid: Tessellate,
) -> list[Distortion]:
    """Read the least distortion of the rows over each tile, and of the night ones.

    Args:
        table: The fixed width table, one row per sample of its product.
        pdsid: The product the table is published beside.
        label: The parsed label every table of the ancillary shares.
        columns: The COLUMN objects of the columns the ancillary reads alone.
        ancillary: What the ancillary is, and which of its columns are read.
        groups: The groups the product still has to be read over, by name.
        grid: The grid the tiles are cut from.

    Returns:
        distortions: One per tile of those groups its rows fall on.
    """
    rows = table.stat().st_size // int(label["ROW_BYTES"])
    read = tables.build_table(table, {**label, "ROWS": str(rows)}, columns)
    flat = grid.flat_tile_indices(
        *grid.tile_indices(read[ancillary.latitude], read[ancillary.longitude])
    )
    order = np.argsort(flat, kind="stable")
    tiles, starts = np.unique(flat[order], return_index=True)
    distortion = read[ancillary.distortion][order]
    night = (read[ancillary.solar_zenith] > ancillary.night_above)[order]
    overall = np.minimum.reduceat(distortion, starts).tolist()
    nightly = np.minimum.reduceat(np.where(night, distortion, np.inf), starts)
    dark = [None if math.isinf(value) else value for value in nightly.tolist()]
    least = dict(zip(tiles.tolist(), zip(dark, overall)))
    distortions: list[Distortion] = []
    for name, group in groups.items():
        for tile in group.tiles:
            pair = least.get(int(grid.flat_tile_indices(tile.band, tile.column)))
            if pair is not None:
                distortions.append(Distortion(name, tile.name, pdsid, *pair))
    return distortions
