"""What the measured dataset holds before the filter is asked of it."""

from __future__ import annotations

import math

import ipywidgets as widgets

from analysis.models.instrument import PIXEL_KM2
from analysis.stats.models import CatalogueStats
from analysis.visualization import panels, wording
from analysis.visualization.panels import Row
from common.maths.physics import METRES_PER_KM

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


def instruments(catalogue: CatalogueStats) -> widgets.Widget:
    """Tabulate what each instrument holds of the measured dataset."""
    rows: list[Row] = []
    for instrument in catalogue.instruments:
        resolution = f"{math.sqrt(PIXEL_KM2[instrument.iid]) * METRES_PER_KM:,.1f} m"
        rows.append(
            (
                instrument.iid,
                f"{instrument.tiles:,}",
                f"{instrument.observations:,}",
                resolution,
                f"{instrument.first:%Y-%m-%d}",
                f"{instrument.last:%Y-%m-%d}",
            )
        )
    return panels.written("Global instrument coverage", _INSTRUMENTS, rows)
