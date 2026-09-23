"""Every drawn tile of the evaluation set on one map, coloured by its class."""

from __future__ import annotations

from collections.abc import Sequence

import ipywidgets as widgets
from matplotlib.lines import Line2D

from analysis.ground_truth.models.label import Label
from analysis.visualization.common import mosaic, panels
from analysis.visualization.dataset.plots import tiles
from common.maths import geodesy

MARKER_SIZE = 18


def plot(labels: Sequence[Label]) -> widgets.Widget:
    """Map every drawn tile at its centre, one colour per class."""
    drawn = [one for one in labels if one.drawn]
    colours = panels.colours(list(dict.fromkeys(one.label for one in drawn)))

    def figure(image: bytes) -> widgets.Widget:
        """Draw the drawn tiles over the mosaic of Mars."""
        board, axis = tiles.mars_board(image, f"{len(drawn):,} tiles drawn")
        for label, colour in colours.items():
            lon, lat = zip(
                *(
                    geodesy.bbox_centre(
                        one.min_lat, one.max_lat, one.west_lon, one.east_lon
                    )
                    for one in drawn
                    if one.label == label
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
            board,
            [
                Line2D([], [], marker="o", linestyle="", color=colour, label=label)
                for label, colour in colours.items()
            ],
        )
        return panels.rendered(board)

    return mosaic.fetched(tiles.MARS, figure, tiles.BASEMAP_PIXELS)
