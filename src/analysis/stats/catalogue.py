"""The measured dataset read as one, before the filter is asked of it."""

from __future__ import annotations

from analysis.coverage import artifacts as index
from analysis.coverage.models.summary import Summary
from analysis.stats.models import CatalogueStats, InstrumentStats, Spread
from common.config import analysis_settings
from common.maths.tessellate import split_bands_columns


def read_catalogue() -> CatalogueStats:
    """Read the grid-wide index as one dataset.

    Returns:
        stats: What it holds, and nothing measured at all when no tile was.
    """
    tile_km = analysis_settings().tile_km
    # One row per tile carries its area, which every set of it shares
    by_tile: dict[str, Summary] = {}
    by_instrument: dict[str, list[Summary]] = {}
    for row in index.catalogued_rows():
        by_tile.setdefault(row.tile, row)
        by_instrument.setdefault(row.iid, []).append(row)
    return CatalogueStats(
        tiles=sum(split_bands_columns(tile_km)),
        tile_km=tile_km,
        measured=len(by_tile),
        tile_km2=Spread.over([row.tile_area_km2 for row in by_tile.values()]),
        instruments=sorted(
            (
                InstrumentStats(
                    iid=iid,
                    tiles=len({row.tile for row in rows}),
                    observations=sum(row.n_obs for row in rows),
                    first=min(row.t_first for row in rows),
                    last=max(row.t_last for row in rows),
                )
                for iid, rows in by_instrument.items()
            ),
            key=lambda instrument: -instrument.observations,
        ),
    )
