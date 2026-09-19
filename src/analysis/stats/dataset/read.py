"""Every tile the selection searched, measured as the dataset stats read it."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

from analysis import configs
from analysis.coverage.artifacts import index
from analysis.selector.models.selection import Selection
from analysis.stats.models.tile import TileStats
from analysis.stats.tile import measure
from analysis.stats.tile import read as tile
from analysis.utils import tile_group
from common.maths.tessellate import Tessellate

# Called with how many tile groups are read and how many there are
Progress = Callable[[int, int], None]


def measure_every_tile(
    picked: Sequence[Selection], workers: int, progress: Progress | None = None
) -> list[TileStats]:
    """Measure what the selection kept of every tile it searched.

    Args:
        picked: What the search left of each tile, as the selection wrote it.
        workers: How many processes to measure on at once, as the run is configured.
        progress: Called with how many tile groups are done and how many there are.

    Returns:
        measured: One entry per tile holding something to measure, a group at a time,
            leaving out a tile with no measured set on disk.
    """
    settings = configs.load()
    grid = Tessellate.of(settings.tile_km)
    by_group: dict[str, list[Selection]] = {}
    for one in picked:
        held = grid.tile_of(one.tile.band, one.tile.column)
        name = tile_group.group_name(len(grid.columns), held, settings.tile_group_deg)
        by_group.setdefault(name, []).append(one)
    found: list[TileStats] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        measured = pool.map(_measure_group, by_group, by_group.values(), chunksize=1)
        for done, stats in enumerate(measured, 1):
            found.extend(stats)
            if progress is not None:
                progress(done, len(by_group))
    return found


def _measure_group(group: str, picked: Sequence[Selection]) -> list[TileStats]:
    """Measure every tile of one group the selection searched.

    Args:
        group: The name of the group to measure.
        picked: What the search left of each of its tiles.

    Returns:
        measured: What the looks each tile keeps left on it, leaving out a tile with no
            measured set on disk or holding nothing measurable.
    """
    coverage = index.load_group(group)
    found: list[TileStats] = []
    for one in picked:
        held = coverage.get(one.tile.tile)
        looks = tile.place_kept_looks(held, one) if held else None
        if looks is not None:
            found.append(measure.measured_tile(looks))
    return found
