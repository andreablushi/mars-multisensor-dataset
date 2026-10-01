"""The stats written out and read back, and the written selection they are read off."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import astuple, fields
from functools import cache
from pathlib import Path

from analysis import paths
from analysis.selector.artifacts import read_selection
from analysis.selector.models.selection import Selection
from analysis.stats.models import DatasetStats, Spread
from common.disk.files import read_json, write_json


def write_stats(stats: DatasetStats, root: Path = paths.STATS_ROOT) -> None:
    """Write out what the filter left of the dataset.

    Args:
        stats: The stats read over every tile searched.
        root: The directory to write it in, made when it is missing.
    """
    laid_out = {field.name: getattr(stats, field.name) for field in fields(stats)}
    write_json(root / paths.STATS_NAME, laid_out, end="\n", indent=1, default=astuple)


def read_stats(root: Path = paths.STATS_ROOT) -> DatasetStats:
    """Read back what the analysis pipeline published.

    Args:
        root: The directory it was written in.

    Returns:
        stats: The stats the run left.
    """
    saved = read_json(root / paths.STATS_NAME)
    return DatasetStats(
        searched=saved["searched"],
        kept=saved["kept"],
        days=Spread(*saved["days"]),
        reached=_spreads_by_iid(saved["reached"]),
        landed_km2_per_look=_spreads_by_iid(saved["landed_km2_per_look"]),
        selected=_spreads_by_iid(saved["selected"]),
        downloads=saved["downloads"],
        overlap=Spread(*saved["overlap"]),
        iids=saved["iids"],
    )


cached_selection = cache(read_selection)


@cache
def selection_by_tile() -> dict[str, Selection]:
    """Read the same selection keyed by the tile each row belongs to, once."""
    return {selection.tile.tile: selection for selection in cached_selection()}


def read_tile_selection(tile: str) -> Selection | None:
    """Read what the selection left of one tile, or None where it holds none."""
    try:
        return selection_by_tile().get(tile)
    except FileNotFoundError:
        return None


def _spreads_by_iid(saved: Mapping[str, Sequence[float]]) -> dict[str, Spread]:
    """Read one measurement per instrument back off the numbers it was written as."""
    return {iid: Spread(*numbers) for iid, numbers in saved.items()}
