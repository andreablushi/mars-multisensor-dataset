"""Every tile the selection searched, measured as the dataset stats read it."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

from analysis import console
from analysis.coverage.artifacts import index
from analysis.selector.models.selection import Selection
from analysis.stats.models.tile import TileStats
from analysis.stats.tile import measure
from analysis.stats.tile import read as tile
from analysis.utils.tile_group import group_of, tile_grid


def measure_every_tile(picked: Sequence[Selection], workers: int) -> list[TileStats]:
    """Measure what the selection kept of every tile it searched.

    Args:
        picked: What the search left of each tile, as the selection wrote it.
        workers: How many processes to measure on at once, as the run is configured.

    Returns:
        measured: One entry per tile with something to measure, a group at a time.
    """
    grid = tile_grid()
    by_group: dict[str, list[Selection]] = {}
    for one in picked:
        name = group_of(grid.tile_of(one.tile.band, one.tile.column))
        by_group.setdefault(name, []).append(one)
    found: list[TileStats] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        measured = pool.map(_measure_group, by_group, by_group.values(), chunksize=1)
        for done, stats in enumerate(measured, 1):
            found.extend(stats)
            console.report("stats", done, len(by_group))
    return found


def _measure_group(group: str, picked: Sequence[Selection]) -> list[TileStats]:
    """Measure every tile of one group the selection searched.

    Args:
        group: The name of the group to measure.
        picked: What the search left of each of its tiles.

    Returns:
        measured: What each tile's kept looks left on it.
    """
    coverage = index.load_group(group)
    found: list[TileStats] = []
    for one in picked:
        held = coverage.get(one.tile.tile)
        looks = tile.place_kept_looks(held, one) if held else None
        if looks is not None:
            found.append(measure.measured_tile(looks))
    return found
