"""SHARAD over one tile as a radargram, as delivered and as the build normalizes it."""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np

from analysis.visualization import panels
from analysis.visualization.preprocessing import product
from building.configs import sharad as configs
from common.models.tile import Tile
from common.pds import images

NAME = "SHARAD"


def plot(identifier: str, frame: Tile) -> widgets.Widget:
    """Show one SHARAD track's traces over a tile, delivered and built."""
    product.fetch_product(NAME, identifier, frame)
    sample = product.tile_sample(
        NAME, product.product_observation(NAME, identifier), frame
    )
    power, _ = images.load_cube(
        configs.CACHE.product_files(identifier, configs.Kind.OBSERVATION)[".img"]
    )
    with np.errstate(divide="ignore"):
        delivered = 10 * np.log10(power[:, sample.traces, 0])
    return panels.side_by_side(
        [
            ("Delivered power (dB)", panels.stretched(delivered)),
            ("Built sample (dB, peak on clutter)", panels.stretched(sample.power)),
        ],
        aspect="auto",
    )
