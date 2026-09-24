"""What the measured dataset holds before the filter is asked of it."""

from __future__ import annotations

import math

import ipywidgets as widgets

from analysis.stats.models import CatalogueStats, DatasetStats
from analysis.visualization import panels, wording
from analysis.visualization.panels import Row

_INSTRUMENTS = (
    "Instrument",
    "Tiles reached",
    "Observations",
    "Resolution",
    "First look",
    "Last look",
)


def measured(catalogue: CatalogueStats) -> widgets.Widget:
    """Tabulate how big the measured dataset is."""
    return panels.written(
        "The ODE dataset that was measured",
        panels.STATISTIC_VALUE,
        [
            ("Tiles Mars is split into", f"{catalogue.tiles:,}"),
            ("Tile side", f"{catalogue.tile_km:,.0f} km"),
            ("Tiles measured", f"{catalogue.measured:,}"),
            ("Mean tile size", wording.spread(catalogue.tile_km2, wording.area)),
        ],
    )


def instruments(catalogue: CatalogueStats, dataset: DatasetStats) -> widgets.Widget:
    """Tabulate what each instrument holds of the measured dataset."""
    rows: list[Row] = []
    for instrument in catalogue.instruments:
        # The median, since a handful of records publish a pixel far out from the rest
        pixel_km2 = dataset.pixel_km2.get(instrument.iid)
        if pixel_km2 is None or not pixel_km2.counted:
            resolution = wording.UNCOUNTED
        else:
            resolution = f"{math.sqrt(pixel_km2.middle) * 1000.0:,.1f} m"
        rows.append(
            (
                instrument.iid,
                f"{instrument.tiles:,}",
                f"{instrument.observations:,}",
                resolution,
                instrument.first.date().isoformat(),
                instrument.last.date().isoformat(),
            )
        )
    return panels.written("Global instrument coverage", _INSTRUMENTS, rows)
