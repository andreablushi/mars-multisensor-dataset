"""The signal phase distortion one SHARAD geometry table holds over each tile."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from analysis.models.ancillary import Distortion
from analysis.models.settings import AnalysisSettings
from analysis.models.tile_group import TileGroup
from building.configs import sharad
from building.preprocessing.sharad.models.observation import (
    LATITUDE_FIELD,
    LONGITUDE_FIELD,
    SOLAR_ZENITH_FIELD,
)
from common.maths.tessellate import Tessellate
from common.pds import tables


def load_distortions(
    table: Path,
    pdsid: str,
    label: dict[str, str],
    columns: list[dict[str, str]],
    settings: AnalysisSettings,
    groups: Mapping[str, TileGroup],
    grid: Tessellate,
) -> list[Distortion]:
    """Read the least distortion of the rows over each tile, and of the night ones.

    Args:
        table: The fixed width table, one row per sample of its product.
        pdsid: The product the table is published beside.
        label: The parsed label every geometry table shares.
        columns: The COLUMN objects of the columns read alone.
        settings: The settled choices for the run, naming the distortion column
            and the night threshold.
        groups: The groups the product still has to be read over, by name.
        grid: The grid the tiles are cut from.

    Returns:
        distortions: One per tile of those groups its rows fall on.
    """
    rows = table.stat().st_size // int(label["ROW_BYTES"])
    read = tables.build_table(table, {**label, "ROWS": str(rows)}, columns)
    flat = grid.flat_tile_indices(
        *grid.tile_indices(read[LONGITUDE_FIELD], read[LATITUDE_FIELD])
    )
    order = np.argsort(flat, kind="stable")
    tiles, starts = np.unique(flat[order], return_index=True)
    distortion = read[settings.sharad_distortion][order]
    night_zenith = settings.criteria.solar_zenith[sharad.LAYOUT.instrument]
    night = (read[SOLAR_ZENITH_FIELD] > night_zenith)[order]
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
