"""How big a dataset the filter leaves, and how much of it a build takes."""

from __future__ import annotations

import ipywidgets as widgets

from analysis.stats.models import DatasetStats
from analysis.visualization import panels, wording
from analysis.visualization.panels import Row


def final(dataset: DatasetStats) -> widgets.Widget:
    """Tabulate the dataset the filter leaves behind."""
    rows: list[Row] = [
        ("Tiles searched", f"{dataset.searched:,}"),
        ("Tiles kept", f"{dataset.kept:,}"),
        ("Mean window", wording.spread(dataset.days, wording.duration)),
    ]
    for iid in dataset.iids:
        selected = dataset.selected[iid]
        rows += [
            (
                f"{iid} observations selected",
                f"{selected.mean * selected.counted:,.0f}",
            ),
            (f"{iid} observations to download", f"{dataset.downloads[iid]:,}"),
        ]
    return panels.written("The dataset the filter leaves", panels.STATISTIC_VALUE, rows)
