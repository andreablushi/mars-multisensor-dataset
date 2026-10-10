"""MOLA around one tile as its sheet publishes it, and as the build cuts it."""

from __future__ import annotations

from dataclasses import replace

import httpx
import ipywidgets as widgets

from analysis.visualization import panels
from analysis.visualization.preprocessing import product
from building.configs import mola as configs
from building.download import mola as download
from building.preprocessing.mola.merge_sheets import merge_sheets
from common.fetch.http import TLS_CONTEXT
from common.maths.geodesy import TURN
from common.models.tile import Tile

NAME = "MOLA"


def plot(sheet: str, frame: Tile) -> widgets.Widget:
    """Show the sheet a tile stands on, three tiles wide, and the tile's built crop."""
    with httpx.Client(verify=TLS_CONTEXT) as client:
        download.fetch_product(
            sheet, configs.NAMING.product(sheet, configs.Kind.TOPOGRAPHY), client
        )
    observation = product.product_observation(NAME, download.mola_grid(frame))
    sample = product.tile_sample(NAME, observation, frame)
    height, span = frame.max_lat - frame.min_lat, frame.span
    around = replace(
        frame,
        min_lat=frame.min_lat - height,
        max_lat=frame.max_lat + height,
        west_lon=(frame.west_lon - span) % TURN,
        east_lon=(frame.east_lon + span) % TURN,
    )
    _, published, _ = merge_sheets(observation, around)
    return panels.side_by_side(
        [
            ("Published sheet, the tile at its centre", panels.stretched(published)),
            ("Built sample", panels.stretched(sample.elevation)),
        ],
        cmap="terrain",
    )
