"""Every tile the selection searched, measured and read as one dataset."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

from analysis import console
from analysis.coverage import artifacts as index
from analysis.selector.models.selection import Selection
from analysis.stats.models import DatasetStats, Spread, TileStats
from analysis.stats.tile import (
    ground_by_instrument_count,
    measured_tile,
    place_kept_looks,
)
from analysis.utils.tile_group import group_of, tile_grid


def measure_every_tile(
    selections: Sequence[Selection], workers: int
) -> list[TileStats]:
    """Measure what the selection kept of every tile it searched.

    Args:
        selections: What the search left of each tile, as the selection wrote it.
        workers: How many processes to measure on at once, as the run is configured.

    Returns:
        measured: One entry per tile with something to measure, a group at a time.
    """
    grid = tile_grid()
    by_group: dict[str, list[Selection]] = {}
    for selection in selections:
        name = group_of(grid.tile_of(selection.tile.band, selection.tile.column))
        by_group.setdefault(name, []).append(selection)
    measured: list[TileStats] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        groups = pool.map(measure_group, by_group, by_group.values(), chunksize=1)
        for done, group_stats in enumerate(groups, 1):
            measured.extend(group_stats)
            console.report("stats", done, len(by_group))
    return measured


def measure_group(group: str, selections: Sequence[Selection]) -> list[TileStats]:
    """Measure every tile of one group the selection searched.

    Args:
        group: The name of the group to measure.
        selections: What the search left of each of its tiles.

    Returns:
        measured: What each tile's kept looks left on it.
    """
    coverage = index.load_group(group)
    measured: list[TileStats] = []
    for selection in selections:
        tile_coverage = coverage.get(selection.tile.tile)
        looks = place_kept_looks(tile_coverage, selection) if tile_coverage else None
        if looks is not None:
            measured.append(measured_tile(looks))
    return measured


def dataset_stats(
    measured: Sequence[TileStats], selections: Sequence[Selection]
) -> DatasetStats:
    """Read every tile the selection searched as one dataset.

    Args:
        measured: What the looks each tile keeps left on it, in any order.
        selections: What the search left of every tile, whose products are counted.

    Returns:
        stats: What the filter left of them.
    """
    iids = list(dict.fromkeys(iid for tile in measured for iid in tile.iids))
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
                [tile.pixel_km2[iid] for tile in measured if iid in tile.pixel_km2]
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
