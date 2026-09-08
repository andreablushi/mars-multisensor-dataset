"""What each instrument lands on a feature, and how far it reaches on it."""

from __future__ import annotations

import ipywidgets as widgets

from analysis.selector import configs as filtering
from analysis.stats.models.dataset import DatasetStats
from analysis.stats.models.spread import Spread
from analysis.visualization.common import quantities, tables, wording
from analysis.visualization.common.models.tables import Row

_LANDED = (
    "Instrument",
    "Mean observations offered",
    "Mean pixels landed per observation",
    "Pixels asked",
    "Mean coverage inside a feature",
    "Least",
)
_BLANK = ""


def landed(read: DatasetStats) -> widgets.Widget:
    """Tabulate what each instrument lands on a feature and how far it reaches."""
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
        "What each instrument lands on a feature and how far it reaches", _LANDED, rows
    )


def _share(measured: Spread) -> tuple[str, str]:
    """Write how much of a feature something reaches.

    Args:
        measured: The share read off every feature that earned a window.

    Returns:
        mean: The mean share with its spread.
        least: The least any feature gave it.
    """
    return (
        wording.spread(measured, lambda share: f"{share:.1%}"),
        f"{measured.low:.1%}",
    )
