"""What each instrument lands on a tile, and how far it reaches on it."""

from __future__ import annotations

import ipywidgets as widgets

from common.analysis.selector import configs as filtering
from common.analysis.stats.models.dataset import DatasetStats
from common.analysis.stats.models.spread import Spread
from common.analysis.visualization.common import quantities, tables, wording
from common.analysis.visualization.common.models.tables import Row

_LANDED = (
    "Instrument",
    "Mean observations offered",
    "Mean pixels landed per observation",
    "Pixels asked",
    "Mean coverage inside a tile",
    "Least",
)
_BLANK = ""


def landed(read: DatasetStats) -> widgets.Widget:
    """Tabulate what each instrument lands on a tile and how far it reaches."""
    rows: list[Row] = []
    for iid in read.iids:
        asked = filtering.FILTER.admits.get(iid)
        # A sounder counts traces, not picture elements, so its pixels go unmarked
        unit = "" if iid == wording.SOUNDER else " px"
        measured = read.held.pixels_per_look[iid]
        rows.append(
            (
                iid,
                wording.spread(read.offered[iid], lambda offered: f"{offered:,.1f}"),
                wording.spread(
                    measured, lambda pixels: f"{quantities.compact(pixels)}{unit}"
                )
                if measured.counted
                else wording.UNCOUNTED,
                f"{asked:,.0f}" if asked else wording.NOTHING,
                *_share(read.held.reached[iid]),
            )
        )
    # The ground no one instrument answers for, so it carries none of their columns
    rows.append(
        (
            f"All {len(read.iids)} instruments overlap",
            _BLANK,
            _BLANK,
            _BLANK,
            *_share(read.overlap),
        )
    )
    return tables.written(
        "What each instrument lands on a tile and how far it reaches", _LANDED, rows
    )


def _share(measured: Spread) -> tuple[str, str]:
    """Write how much of a tile something reaches.

    Args:
        measured: The share read off every tile that earned a window.

    Returns:
        mean: The mean share with its spread.
        least: The least any tile gave it.
    """
    return (
        wording.spread(measured, lambda share: f"{share:.1%}"),
        f"{measured.low:.1%}",
    )
