"""Loading the signal phase distortion one ancillary table holds over each group."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from analysis.ground_truth import box
from analysis.models.ancillary import Ancillary, Distortion
from analysis.models.tile_group import TileGroup
from building.common.pds import tables


def load_distortions(
    table: Path,
    pdsid: str,
    label: dict[str, str],
    columns: list[dict[str, str]],
    ancillary: Ancillary,
    groups: Mapping[str, TileGroup],
) -> list[Distortion]:
    """Read the least distortion of the rows inside each group, and of the night ones.

    Args:
        table: The fixed width table, one row per sample of its product.
        pdsid: The product the table is published beside.
        label: The parsed label every table of the ancillary shares.
        columns: The COLUMN objects of the columns the ancillary reads alone.
        ancillary: What the ancillary is, and which of its columns are read.
        groups: The groups the product still has to be read over, by name.

    Returns:
        distortions: One per group.
    """
    rows = table.stat().st_size // int(label["ROW_BYTES"])
    read = tables.build_table(table, {**label, "ROWS": str(rows)}, columns)
    latitude = read[ancillary.latitude]
    point = (latitude, latitude, read[ancillary.longitude] % 360.0, 0.0)
    distortion = read[ancillary.distortion]
    night = read[ancillary.solar_zenith] > ancillary.night_above
    distortions: list[Distortion] = []
    for name, group in groups.items():
        inside = box.inside(point, box.bounds_box(group))
        least = [
            float(distortion[counted].min()) if counted.any() else None
            for counted in (inside & night, inside)
        ]
        distortions.append(Distortion(name, pdsid, *least))
    return distortions
