"""What each instrument lands on a tile, and how far it reaches on it."""

from __future__ import annotations

import ipywidgets as widgets

from analysis import configs
from analysis.stats.models.dataset import DatasetStats
from analysis.stats.models.spread import Spread
from analysis.visualization.common import quantities, tables, wording
from analysis.visualization.common.models.tables import Row

_LANDED = (
    "Instrument",
    "Mean observations selected",
    "Mean pixels landed per observation",
    "Pixels asked",
    "Mean coverage inside a tile",
)
_BLANK = ""
_WHOLE = "{:,.0f}".format


def landed(read: DatasetStats) -> widgets.Widget:
    """Tabulate what each instrument lands on a tile and how far it reaches."""
    rows: list[Row] = []
    admits = configs.load().window.admits
    for iid in read.iids:
        asked = admits.get(iid)
        # A sounder counts traces, not picture elements, so its pixels go unmarked
        unit = "" if iid == wording.SOUNDER else " px"
        measured = read.held.pixels_per_look[iid]
        selected = read.selected[iid]
        rows.append(
            (
                iid,
                wording.spread(selected, "{:,.1f}".format, _WHOLE),
                wording.spread(
                    measured, lambda pixels: f"{quantities.compact(pixels)}{unit}"
                )
                if measured.counted
                else wording.UNCOUNTED,
                f"{asked:,.0f}" if asked else wording.NOTHING,
                _share(read.held.reached[iid]),
            )
        )
    # The ground no one instrument answers for, so it carries none of their columns
    rows.append(
        (
            f"All {len(read.iids)} instruments overlap",
            _BLANK,
            _BLANK,
            _BLANK,
            _share(read.overlap),
        )
    )
    return tables.written(
        "What each instrument lands on a tile and how far it reaches", _LANDED, rows
    )


def _share(measured: Spread) -> str:
    """Write how much of a tile something reaches.

    Args:
        measured: The share read off every tile that earned a window.

    Returns:
        share: The mean share, then the least and the most any tile gave it.
    """
    return wording.spread(measured, lambda share: f"{share:.1%}")
