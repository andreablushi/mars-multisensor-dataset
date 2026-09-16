"""What the measured dataset holds before the filter is asked of it."""

from __future__ import annotations

import math

import ipywidgets as widgets

from analysis.stats.models.catalogue import CatalogueStats
from analysis.stats.models.dataset import DatasetStats
from analysis.visualization.common import quantities, tables, wording
from analysis.visualization.common.models.tables import Row

_TILES = ("Statistic", "Value")
_INSTRUMENTS = (
    "Instrument",
    "Tiles reached",
    "Observations",
    "Resolution",
    "First look",
    "Last look",
)


def measured(stats: CatalogueStats) -> widgets.Widget:
    """Tabulate how big the measured dataset is."""
    return tables.written(
        "The ODE dataset that was measured",
        _TILES,
        [
            ("Tiles Mars is split into", f"{stats.tiles:,}"),
            ("Tile side", f"{stats.tile_km:,.0f} km"),
            ("Tiles measured", f"{stats.measured:,}"),
            ("Mean tile size", wording.spread(stats.tile_km2, quantities.area)),
        ],
    )


def instruments(stats: CatalogueStats, read: DatasetStats) -> widgets.Widget:
    """Tabulate what each instrument holds of the measured dataset."""
    rows: list[Row] = []
    for instrument in stats.instruments:
        # The median, since a handful of records publish a pixel far out from the rest
        pixel_km2 = read.held.pixel_km2.get(instrument.iid)
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
    return tables.written("Global instrument coverage", _INSTRUMENTS, rows)
