"""What each instrument lands on a tile, and how far it reaches on it."""

from __future__ import annotations

import ipywidgets as widgets

from analysis.stats.models import DatasetStats
from analysis.visualization import panels, wording
from analysis.visualization.panels import Row
from common.config import analysis_settings

_LANDED = (
    "Instrument",
    "Mean observations selected",
    "Mean pixels landed per observation",
    "Pixels asked",
    "Mean coverage inside a tile",
)


def landed(dataset: DatasetStats) -> widgets.Widget:
    """Tabulate what each instrument lands on a tile and how far it reaches."""
    rows: list[Row] = []
    admits = analysis_settings().window.admits
    percent = "{:.1%}".format
    for iid in dataset.iids:
        # A sounder counts traces, not picture elements, so its pixels go unmarked
        unit = "" if iid == wording.SOUNDER else " px"
        pixels = dataset.pixels_per_look[iid]
        if pixels.counted:
            pixels_landed = wording.spread(
                pixels, lambda count: f"{wording.compact(count)}{unit}"
            )
        else:
            pixels_landed = wording.UNCOUNTED
        asked = admits.get(iid)
        rows.append(
            (
                iid,
                wording.spread(
                    dataset.selected[iid], "{:,.1f}".format, "{:,.0f}".format
                ),
                pixels_landed,
                f"{asked:,.0f}" if asked else wording.NOTHING,
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
        "What each instrument lands on a tile and how far it reaches", _LANDED, rows
    )
