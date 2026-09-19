"""What the stats made of the dataset, written out so they need not be read again."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from analysis import paths
from analysis.stats.models.dataset import Aggregate, DatasetStats
from analysis.stats.models.spread import Spread
from common.disk.files import atomic_path

# The layout of a published file, raised whenever what is written changes.
STATS_SHAPE = 3


def stats_path(root: Path = paths.STATS_ROOT) -> Path:
    """Return the file the stats are published as.

    Args:
        root: The directory it is written in.

    Returns:
        path: The file, which need not exist.
    """
    return root / paths.STATS_NAME


def write_stats_file(held: DatasetStats, root: Path = paths.STATS_ROOT) -> Path:
    """Write out what the filter left of the dataset.

    Args:
        held: The stats read over every tile searched.
        root: The directory to write it in, made when it is missing.

    Returns:
        path: The file written.
    """
    aggregate = held.held
    laid_out = {
        "shape": STATS_SHAPE,
        "iids": held.iids,
        "held": {
            "searched": aggregate.searched,
            "kept": aggregate.kept,
            "days": _numbers_of(aggregate.days),
            "reached": _numbers_by_iid(aggregate.reached),
            "per_look": _numbers_by_iid(aggregate.pixels_per_look),
            "pixel_km2": _numbers_by_iid(aggregate.pixel_km2),
        },
        "offered": _numbers_by_iid(held.offered),
        "overlap": _numbers_of(held.overlap),
    }
    path = stats_path(root)
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(laid_out, indent=1) + "\n", encoding="utf-8")
    return path


def read_stats_file(root: Path = paths.STATS_ROOT) -> DatasetStats:
    """Read back what the stats pipeline published.

    Args:
        root: The directory it was written in.

    Returns:
        stats: The stats the run left.

    Raises:
        ValueError: When the file was written in another layout.
    """
    path = stats_path(root)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["shape"] != STATS_SHAPE:
        raise ValueError(f"{path.name} holds shape {saved['shape']}, not {STATS_SHAPE}")
    held = saved["held"]
    return DatasetStats(
        held=Aggregate(
            searched=held["searched"],
            kept=held["kept"],
            days=_spread_of(held["days"]),
            reached=_spreads_by_iid(held["reached"]),
            pixels_per_look=_spreads_by_iid(held["per_look"]),
            pixel_km2=_spreads_by_iid(held["pixel_km2"]),
        ),
        offered=_spreads_by_iid(saved["offered"]),
        overlap=_spread_of(saved["overlap"]),
        iids=saved["iids"],
    )


def _numbers_by_iid(measured: Mapping[str, Spread]) -> dict[str, list[float]]:
    """Write out one measurement per instrument.

    Args:
        measured: The measurement each instrument left, by instrument.

    Returns:
        numbers: The numbers each of them holds, by instrument.
    """
    return {iid: _numbers_of(one) for iid, one in measured.items()}


def _spreads_by_iid(saved: Mapping[str, Sequence[float]]) -> dict[str, Spread]:
    """Read one measurement per instrument back.

    Args:
        saved: The numbers each instrument's measurement was written as.

    Returns:
        spreads: The measurement each of them left, by instrument.
    """
    return {iid: _spread_of(one) for iid, one in saved.items()}


def _numbers_of(measured: Spread) -> list[float]:
    """Write one measurement out as the numbers it holds.

    Args:
        measured: The measurement read off many tiles.

    Returns:
        numbers: Its numbers, in the order the spread names them.
    """
    return [
        measured.mean,
        measured.middle,
        measured.deviation,
        measured.low,
        measured.high,
        measured.counted,
    ]


def _spread_of(saved: Sequence[float]) -> Spread:
    """Read one measurement back off the numbers it was written as.

    Args:
        saved: Its numbers, in the order the spread names them.

    Returns:
        spread: The measurement.
    """
    mean, middle, deviation, low, high, counted = saved
    return Spread(mean, middle, deviation, low, high, int(counted))
