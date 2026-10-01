"""Every drawn tile of the evaluation set on one map, coloured by its class."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import ipywidgets as widgets
from matplotlib.lines import Line2D

from analysis.ground_truth.models.label import Label
from analysis.ground_truth.models.settings import GroundTruthSettings
from analysis.visualization import mosaic, panels
from analysis.visualization.ground_truth.drawn import NOTHING_DRAWN, drawn_by_class
from common.maths import geodesy

MARKER_SIZE = 18


def plot(labels: Sequence[Label], settings: GroundTruthSettings) -> widgets.Widget:
    """Map every drawn tile at its centre, one colour per class."""
    grouped = drawn_by_class(labels, settings)
    if not grouped:
        return panels.unavailable(NOTHING_DRAWN)
    return mosaic.fetched(
        mosaic.MARS, partial(classes_map, grouped), mosaic.MARS_PIXELS
    )


def classes_map(grouped: dict[str, list[Label]], image: bytes) -> widgets.Image:
    """Draw the drawn tiles over the mosaic of Mars, one colour per class.

    Args:
        grouped: The tiles the balanced draw took, by class in config order.
        image: The mosaic of the whole planet, as fetched.

    Returns:
        map: The map, rendered.
    """
    drawn = sum(len(tiles) for tiles in grouped.values())
    figure, axis = mosaic.mars_board(image, f"{drawn:,} tiles drawn")
    colours = panels.colours(list(grouped))
    for name, tiles in grouped.items():
        lon, lat = zip(*(geodesy.bbox_centre(label) for label in tiles), strict=True)
        axis.scatter(
            lon,
            lat,
            s=MARKER_SIZE,
            color=colours[name],
            edgecolor="black",
            linewidth=0.4,
            transform=mosaic.LONLAT,
        )
    panels.key_beside(
        figure,
        [
            Line2D([], [], marker="o", linestyle="", color=colour, label=name)
            for name, colour in colours.items()
        ],
    )
    return panels.rendered(figure)
