"""Every drawn tile of the evaluation set on one map, coloured by its class."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import ipywidgets as widgets
from matplotlib.lines import Line2D

from analysis.ground_truth.models.label import Label
from analysis.visualization import mosaic, panels
from analysis.visualization.dataset import tiles
from analysis.visualization.panels import Colour
from common.maths import geodesy

MARKER_SIZE = 18


def plot(labels: Sequence[Label]) -> widgets.Widget:
    """Map every drawn tile at its centre, one colour per class."""
    drawn = [label for label in labels if label.drawn]
    colours = panels.colours(list(dict.fromkeys(label.label for label in drawn)))
    return mosaic.fetched(
        tiles.MARS, partial(classes_map, drawn, colours), tiles.BASEMAP_PIXELS
    )


def classes_map(
    drawn: Sequence[Label], colours: dict[str, Colour], image: bytes
) -> widgets.Image:
    """Draw the drawn tiles over the mosaic of Mars, one colour per class.

    Args:
        drawn: The tiles the balanced draw took.
        colours: The colour each class is drawn in.
        image: The mosaic of the whole planet, as fetched.

    Returns:
        map: The map, rendered.
    """
    figure, axis = tiles.mars_board(image, f"{len(drawn):,} tiles drawn")
    for name, colour in colours.items():
        lon, lat = zip(
            *(
                geodesy.bbox_centre(
                    label.min_lat, label.max_lat, label.west_lon, label.east_lon
                )
                for label in drawn
                if label.label == name
            ),
            strict=True,
        )
        axis.scatter(
            lon,
            lat,
            s=MARKER_SIZE,
            color=colour,
            edgecolor="black",
            linewidth=0.4,
            transform=tiles.LONLAT,
        )
    panels.key_beside(
        figure,
        [
            Line2D([], [], marker="o", linestyle="", color=colour, label=name)
            for name, colour in colours.items()
        ],
    )
    return panels.rendered(figure)
