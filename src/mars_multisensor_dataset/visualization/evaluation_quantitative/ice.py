"""What the SWIM ice maps come to on every tile, over the mosaic of Mars."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from functools import partial

import ipywidgets as widgets
import numpy as np
from matplotlib.lines import Line2D

from mars_multisensor_dataset.analysis.ground_truth.swim.models import IceLabel
from mars_multisensor_dataset.visualization import mosaic, panels
from mars_multisensor_dataset.visualization.config import MARS_PIXELS
from mars_multisensor_dataset.visualization.evaluation_quantitative import config


def tile_category(label: IceLabel) -> str:
    """Return which depths one tile is icy at, as the key names it."""
    shallow, deep = label.ice_1_5m, label.ice_5m
    return config.CATEGORIES[
        0 if shallow and deep else 1 if shallow else 2 if deep else 3
    ]


def plot_tiles(labels: Sequence[IceLabel]) -> widgets.Widget:
    """Map every labelled tile at its centre, coloured by the depths it is icy at."""
    return mosaic.fetched(mosaic.MARS, partial(tiles_map, labels), MARS_PIXELS)


def tiles_map(labels: Sequence[IceLabel], image: bytes) -> widgets.Image:
    """Draw every labelled tile over the mosaic of Mars, one colour per category.

    Args:
        labels: Every tile the SWIM maps labelled.
        image: The mosaic of the whole planet, as fetched.

    Returns:
        map: The map, rendered.
    """
    figure, axis = mosaic.mars_board(image, f"{len(labels):,} tiles labelled")
    colours = panels.colours(list(config.CATEGORIES))
    categories = np.array([tile_category(label) for label in labels])
    lon = np.array([label.centre_lon for label in labels])
    lat = np.array([label.centre_lat for label in labels])
    counts = Counter(categories)
    for name, colour in colours.items():
        held = categories == name
        axis.scatter(
            lon[held],
            lat[held],
            s=config.TILE_MARKER,
            color=colour,
            linewidth=0,
            transform=mosaic.LONLAT,
        )
    panels.key_beside(
        figure,
        [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=colour,
                label=f"{name} ({counts[name]:,})",
            )
            for name, colour in colours.items()
        ],
    )
    return panels.rendered(figure)
