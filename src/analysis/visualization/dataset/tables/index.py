"""What the measured dataset holds before the filter is asked of it."""

from __future__ import annotations

import math

import ipywidgets as widgets

from analysis.stats.models.catalogue import CatalogueStats
from analysis.stats.models.dataset import DatasetStats
from analysis.visualization.common import tables, wording

_FEATURES = ("Statistic", "Value")
_INSTRUMENTS = (
    "Instrument",
    "Features reached",
    "Observations",
    "Resolution",
    "First look",
    "Last look",
)


def measured(stats: CatalogueStats) -> widgets.Widget:
    """Tabulate how big the measured dataset is."""
    return tables.written(
        "The ODE dataset that was measured",
        _FEATURES,
        [
            ("Features catalogued", f"{stats.catalogued:,}"),
            ("Features dropped as points", f"{stats.points:,}"),
            ("Features measured", f"{stats.features:,}"),
            ("Feature classes", f"{len(stats.classes):,}"),
        ],
    )


def instruments(stats: CatalogueStats, read: DatasetStats) -> widgets.Widget:
    """Tabulate what each instrument holds of the measured dataset."""
    return tables.written(
        "Global instrument coverage",
        _INSTRUMENTS,
        [
            (
                instrument.iid,
                f"{instrument.features:,}",
                f"{instrument.observations:,}",
                _resolution(read, instrument.iid),
                instrument.first.date().isoformat(),
                instrument.last.date().isoformat(),
            )
            for instrument in stats.instruments
        ],
    )


def _resolution(read: DatasetStats, iid: str) -> str:
    """Write how wide the ground one pixel of an instrument covers is.

    Args:
        read: What the filter left of the dataset, holding the pixel sizes.
        iid: The instrument to write it for.

    Returns:
        written: The side of that ground in metres, or that it was never measured.
    """
    # The median, since a handful of records publish a pixel far out from the rest
    measured = read.held.pixel_km2.get(iid)
    if measured is None or not measured.counted:
        return wording.UNCOUNTED
    return f"{math.sqrt(measured.middle) * 1000.0:,.1f} m"
