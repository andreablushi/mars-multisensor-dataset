"""The tables of a dataset the filter leaves: what it lands, and how big it is."""

from __future__ import annotations

import ipywidgets as widgets

from analysis.stats.models import DatasetStats
from common.config import analysis_settings
from visualization import config, panels, wording
from visualization.panels import Row


def landed(dataset: DatasetStats) -> widgets.Widget:
    """Tabulate what each instrument lands on a tile and how far it reaches."""
    rows: list[Row] = []
    admits = analysis_settings().criteria.admits
    percent = "{:.1%}".format
    for iid in dataset.iids:
        landed_km2 = dataset.landed_km2_per_look[iid]
        if landed_km2.counted:
            # A sounder counts traces, not picture elements
            pixels_landed = wording.spread(
                landed_km2, lambda km2: wording.pixels_of_ground(km2, iid)
            )
        else:
            pixels_landed = config.UNCOUNTED
        asked = admits.get(iid)
        rows.append(
            (
                iid,
                wording.spread(
                    dataset.selected[iid], "{:,.1f}".format, "{:,.0f}".format
                ),
                pixels_landed,
                wording.pixels(asked, iid) if asked else config.NOTHING,
                wording.spread(dataset.reached[iid], percent),
            )
        )
    # The ground no one instrument answers for, so it carries none of their columns
    rows.append(
        (
            f"All {len(dataset.iids)} instruments overlap",
            "",
            "",
            "",
            wording.spread(dataset.overlap, percent),
        )
    )
    return panels.written(
        "What each instrument lands on a tile and how far it reaches",
        config.LANDED,
        rows,
    )


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
    return panels.written("The dataset the filter leaves", config.STATISTIC_VALUE, rows)
