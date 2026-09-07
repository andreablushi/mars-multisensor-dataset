"""What the stats made of the dataset, written out so they need not be read again."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import utils.disk.paths as paths
from analysis.stats import configs
from analysis.stats.models.dataset import Aggregate, ClassStats, DatasetStats
from analysis.stats.models.spread import Spread
from utils.disk.files import atomic_path


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
        held: The stats read over every feature searched.
        root: The directory to write it in, made when it is missing.

    Returns:
        path: The file written.
    """
    path = stats_path(root)
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(_as_json(held), indent=1) + "\n", encoding="utf-8")
    return path


def read_stats_file(root: Path = paths.STATS_ROOT) -> DatasetStats:
    """Read back what the stats pipeline published."""
    path = stats_path(root)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved["shape"] != configs.STATS_SHAPE:
        raise ValueError(
            f"{path.name} holds shape {saved['shape']}, not {configs.STATS_SHAPE}"
        )
    return _from_json(saved)


def _as_json(stats: DatasetStats) -> dict[str, Any]:
    """Lay the stats out as the file holds them.

    Args:
        stats: What the filter left of the dataset.

    Returns:
        held: What the file is written from.
    """
    held = stats.held
    return {
        "shape": configs.STATS_SHAPE,
        "classes": {
            name: [held.selected, _spreads(held.taken)]
            for name, held in stats.classes.items()
        },
        "iids": stats.iids,
        "held": {
            "searched": held.searched,
            "kept": held.kept,
            "days": _spread(held.days),
            "reached": _spreads(held.reached),
            "per_look": _spreads(held.pixels_per_look),
            "pixel_km2": _spreads(held.pixel_km2),
        },
        "offered": _spreads(stats.offered),
        "overlap": _spread(stats.overlap),
    }


def _from_json(saved: Mapping[str, Any]) -> DatasetStats:
    """Read the stats back off what the file holds.

    Args:
        saved: What the file was written with.

    Returns:
        stats: The stats the run left.

    """
    held = saved["held"]
    return DatasetStats(
        classes={
            name: ClassStats(int(selected), _spreads_back(taken))
            for name, (selected, taken) in saved["classes"].items()
        },
        held=Aggregate(
            searched=held["searched"],
            kept=held["kept"],
            days=_read(held["days"]),
            reached=_spreads_back(held["reached"]),
            pixels_per_look=_spreads_back(held["per_look"]),
            pixel_km2=_spreads_back(held["pixel_km2"]),
        ),
        offered=_spreads_back(saved["offered"]),
        overlap=_read(saved["overlap"]),
        iids=saved["iids"],
    )


def _spreads(measured: Mapping[str, Spread]) -> dict[str, list[float]]:
    """Write out one measurement per instrument.

    Args:
        measured: The measurement each instrument left, by instrument.

    Returns:
        numbers: The numbers each of them holds, by instrument.
    """
    return {iid: _spread(one) for iid, one in measured.items()}


def _spreads_back(saved: Mapping[str, Sequence[float]]) -> dict[str, Spread]:
    """Read one measurement per instrument back.

    Args:
        saved: The numbers each instrument's measurement was written as.

    Returns:
        spreads: The measurement each of them left, by instrument.
    """
    return {iid: _read(one) for iid, one in saved.items()}


def _spread(measured: Spread) -> list[float]:
    """Write one measurement out as the numbers it holds.

    Args:
        measured: The measurement read off many features.

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


def _read(saved: Sequence[float]) -> Spread:
    """Read one measurement back off the numbers it was written as.

    Args:
        saved: Its numbers, in the order the spread names them.

    Returns:
        spread: The measurement.
    """
    mean, middle, deviation, low, high, counted = saved
    return Spread(mean, middle, deviation, low, high, int(counted))
