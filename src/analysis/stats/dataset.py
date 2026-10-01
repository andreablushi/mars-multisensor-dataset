"""Every tile the selection searched and measured, read as one dataset."""

from __future__ import annotations

from collections.abc import Sequence

from analysis.selector.models.selection import Selection
from analysis.stats.models import DatasetStats, Spread, TileStats
from analysis.stats.order import config_rank
from analysis.stats.tile import ground_by_instrument_count


def dataset_stats(
    measured: Sequence[TileStats], selections: Sequence[Selection]
) -> DatasetStats:
    """Read every tile the selection searched as one dataset.

    Args:
        measured: What the observations each tile keeps left on it, in any order.
        selections: What the search left of every tile, whose products are counted.

    Returns:
        stats: What the filter left of them.
    """
    iids = sorted(
        dict.fromkeys(iid for tile in measured for iid in tile.iids), key=config_rank
    )
    kept = [tile for tile in measured if tile.window.kept]
    kept_names = {tile.window.tile for tile in kept}
    # A product landing on many tiles is still downloaded once
    products: dict[str, set[str]] = {iid: set() for iid in iids}
    for selection in selections:
        if selection.tile.tile in kept_names:
            for observation in selection.observations:
                if observation.iid in products:
                    products[observation.iid].add(observation.pdsid)
    return DatasetStats(
        searched=len(measured),
        kept=len(kept),
        days=Spread.over([tile.window.days for tile in kept]),
        reached={
            iid: Spread.over(
                [
                    tile.reached[iid].km2 / tile.window.area_km2
                    if iid in tile.reached
                    else 0.0
                    for tile in kept
                ]
            )
            for iid in iids
        },
        landed_km2_per_look={
            iid: Spread.over(
                [
                    tile.reached[iid].landed_km2_per_look
                    for tile in kept
                    if iid in tile.reached
                ]
            )
            for iid in iids
        },
        selected={
            iid: Spread.over(
                [
                    tile.reached[iid].observations_taken if iid in tile.reached else 0
                    for tile in kept
                ]
            )
            for iid in iids
        },
        downloads={iid: len(pdsids) for iid, pdsids in products.items()},
        # The share of a tile every instrument at once reaches, one by one
        overlap=Spread.over(
            [
                ground_by_instrument_count(tile.overlaps).get(len(iids), 0.0)
                / tile.window.area_km2
                for tile in kept
            ]
        ),
        iids=iids,
    )
