"""How big a dataset the filter leaves, and how much of it a build takes."""

from __future__ import annotations

import ipywidgets as widgets

from analysis.stats.models.dataset import DatasetStats
from analysis.visualization.common import quantities, tables, wording
from analysis.visualization.common.models.tables import Row

_HEADINGS = ("Statistic", "Value")


def final(read: DatasetStats) -> widgets.Widget:
    """Tabulate the dataset the filter leaves behind."""
    held = read.held
    rows: list[Row] = [
        ("Tiles searched", f"{held.searched:,}"),
        ("Tiles kept", f"{held.kept:,}"),
        ("Mean window", wording.spread(held.days, quantities.duration)),
        ("Longest window", quantities.duration(held.days.high)),
    ]
    for iid in read.iids:
        measured = read.selected[iid]
        rows.append(
            (
                f"{iid} observations selected",
                f"{measured.mean * measured.counted:,.0f}",
            )
        )
    return tables.written("The dataset the filter leaves", _HEADINGS, rows)
