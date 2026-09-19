"""Every drawn tile of the evaluation set on one map, coloured by its class."""

from __future__ import annotations

from collections.abc import Sequence

import ipywidgets as widgets
from matplotlib.lines import Line2D

from common.analysis.selector.models.selection import Selection
from common.analysis.visualization.common import mosaic, panels
from common.analysis.visualization.dataset.plots import tiles
from common.maths import geodesy
from evaluation.analysis.models.label import Label

MARKER_SIZE = 18


def plot(picked: Sequence[Selection], labels: Sequence[Label]) -> widgets.Widget:
    """Map every drawn tile at its centre, one colour per class."""
    boxes = {one.tile.tile: one.tile for one in picked}
    drawn = [one for one in labels if one.drawn and one.tile in boxes]
    colours = panels.colours(list(dict.fromkeys(one.label for one in drawn)))

    def figure(image: bytes) -> widgets.Widget:
        """Draw the drawn tiles over the mosaic of Mars."""
        board, axis = tiles.mars_board(image, f"{len(drawn):,} tiles drawn")
        for label, colour in colours.items():
            lon, lat = zip(
                *(
                    geodesy.bbox_centre(
                        held.min_lat, held.max_lat, held.west_lon, held.east_lon
                    )
                    for held in (boxes[one.tile] for one in drawn if one.label == label)
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
