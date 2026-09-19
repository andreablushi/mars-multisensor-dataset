"""Reading a run of tiles as one dataset, under the one filter."""

from __future__ import annotations

from collections.abc import Sequence

from common.analysis.stats.models.dataset import Aggregate, DatasetStats
from common.analysis.stats.models.spread import Spread
from common.analysis.stats.models.tile import TileStats
from common.analysis.stats.tile import measure

# How far past the whole tile a share may read before it is thrown out.
SHARE_CEILING = 1.01


def dataset_stats(measured: Sequence[TileStats]) -> DatasetStats:
    """Read every tile the selection searched as one dataset.

    Args:
        measured: What the looks each tile keeps left on it, in any order.

    Returns:
        stats: What the filter left of them.
    """
    iids = list(dict.fromkeys(iid for one in measured for iid in one.iids))
    # A tile claiming more ground than it holds is left out of every figure
    held = [one for one in measured if plausible(one)]
    grounded = [one for one in held if one.window.kept]
    return DatasetStats(
        held=aggregate_tiles(held, iids),
        offered={
            iid: Spread.over([one.offered.get(iid, 0) for one in held]) for iid in iids
        },
        # The share of a tile every instrument at once reaches, one by one
        overlap=Spread.over(
            [
                measure.ground_by_instrument_count(one.overlaps).get(len(iids), 0.0)
                / one.window.area_km2
                for one in grounded
            ]
        ),
        iids=iids,
    )


def aggregate_tiles(measured: Sequence[TileStats], iids: Sequence[str]) -> Aggregate:
    """Read a run of tiles as one.

    Args:
        measured: The tiles something readable was left on, in any order.
        iids: The instruments to report on, in the order to report them.

    Returns:
        held: What they hold between them.
    """
    kept = [one for one in measured if one.window.kept]
    return Aggregate(
        searched=len(measured),
        kept=len(kept),
        days=Spread.over([one.window.days for one in kept]),
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
        pixels_per_look={
            iid: Spread.over(
                [
                    tile.reached[iid].pixels_per_look
                    for tile in kept
                    if iid in tile.reached
                    and tile.reached[iid].pixels_per_look is not None
                ]
            )
            for iid in iids
        },
        # A pixel is the same size wherever it falls, so every searched tile says
        pixel_km2={
            iid: Spread.over(
                [one.pixel_km2[iid] for one in measured if iid in one.pixel_km2]
            )
            for iid in iids
        },
    )


def plausible(tile: TileStats) -> bool:
    """Say whether a tile reports no more ground than it holds.

    Args:
        tile: One tile the search ran over.

    Returns:
        plausible: Whether every share it reports sits inside the ceiling.
    """
    area_km2 = tile.window.area_km2
    shares = [reach.km2 / area_km2 for reach in tile.reached.values()]
    shares.append(sum(tile.overlaps.values()) / area_km2)
    shares.append(tile.window.geo_mean)
    return max(shares) <= SHARE_CEILING
