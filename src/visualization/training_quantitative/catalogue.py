"""What the measured dataset holds before the filter is asked of it."""

from __future__ import annotations

import math

import ipywidgets as widgets

from analysis.models.instrument import PIXEL_KM2
from analysis.stats.models import CatalogueStats
from common.maths.physics import METRES_PER_KM
from visualization import panels, wording
from visualization.config import STATISTIC_VALUE
from visualization.training_quantitative import config


def measured(catalogue: CatalogueStats) -> widgets.Widget:
    """Tabulate how big the measured dataset is."""
    return panels.written(
        "The ODE dataset that was measured",
        STATISTIC_VALUE,
        [
            ("Tiles Mars is split into", f"{catalogue.tiles:,}"),
            ("Tile side", f"{catalogue.tile_km:,.0f} km"),
            ("Tiles measured", f"{catalogue.measured:,}"),
            ("Mean tile size", wording.spread(catalogue.tile_km2, wording.area)),
        ],
    )


def instruments(catalogue: CatalogueStats) -> widgets.Widget:
    """Tabulate what each instrument holds of the measured dataset."""
    rows = [
        (
            instrument.iid,
            f"{instrument.tiles:,}",
            f"{instrument.observations:,}",
            f"{math.sqrt(PIXEL_KM2[instrument.iid]) * METRES_PER_KM:,.1f} m",
            f"{instrument.first:%Y-%m-%d}",
            f"{instrument.last:%Y-%m-%d}",
        )
        for instrument in catalogue.instruments
    ]
    return panels.written("Global instrument coverage", config.INSTRUMENTS, rows)
