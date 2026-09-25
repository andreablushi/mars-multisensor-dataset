"""The stats written out and read back, and the written selection they are read off."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import astuple
from functools import cache
from pathlib import Path

from analysis import paths
from analysis.selector.artifacts import read_selection
from analysis.selector.models.selection import Selection
from analysis.stats.models import DatasetStats, Spread
from common.disk.files import atomic_path


def write_stats(stats: DatasetStats, root: Path = paths.STATS_ROOT) -> Path:
    """Write out what the filter left of the dataset.

    Args:
        stats: The stats read over every tile searched.
        root: The directory to write it in, made when it is missing.

    Returns:
        path: The file written.
    """
    laid_out = {
        "searched": stats.searched,
        "kept": stats.kept,
        "days": astuple(stats.days),
        "reached": _numbers_by_iid(stats.reached),
        "pixels_per_look": _numbers_by_iid(stats.pixels_per_look),
        "pixel_km2": _numbers_by_iid(stats.pixel_km2),
        "selected": _numbers_by_iid(stats.selected),
        "downloads": stats.downloads,
        "overlap": astuple(stats.overlap),
        "iids": stats.iids,
    }
    path = root / paths.STATS_NAME
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(laid_out, indent=1) + "\n", encoding="utf-8")
    return path


def read_stats(root: Path = paths.STATS_ROOT) -> DatasetStats:
    """Read back what the stats pipeline published.

    Args:
        root: The directory it was written in.

    Returns:
        stats: The stats the run left.
    """
    saved = json.loads((root / paths.STATS_NAME).read_text(encoding="utf-8"))
    return DatasetStats(
        searched=saved["searched"],
        kept=saved["kept"],
        days=Spread(*saved["days"]),
        reached=_spreads_by_iid(saved["reached"]),
        pixels_per_look=_spreads_by_iid(saved["pixels_per_look"]),
        pixel_km2=_spreads_by_iid(saved["pixel_km2"]),
        selected=_spreads_by_iid(saved["selected"]),
        downloads=saved["downloads"],
        overlap=Spread(*saved["overlap"]),
        iids=saved["iids"],
    )


@cache
def cached_selection() -> list[Selection]:
    """Read what the selection left of every tile it searched, once."""
    return read_selection()


@cache
def selection_by_tile() -> dict[str, Selection]:
    """Read the same selection keyed by the tile each row belongs to, once."""
    return {selection.tile.tile: selection for selection in cached_selection()}


def _numbers_by_iid(spreads: Mapping[str, Spread]) -> dict[str, tuple[float, ...]]:
    """Write out one measurement per instrument as the numbers it holds."""
    return {iid: astuple(spread) for iid, spread in spreads.items()}


def _spreads_by_iid(saved: Mapping[str, Sequence[float]]) -> dict[str, Spread]:
    """Read one measurement per instrument back off the numbers it was written as."""
    return {iid: Spread(*numbers) for iid, numbers in saved.items()}
