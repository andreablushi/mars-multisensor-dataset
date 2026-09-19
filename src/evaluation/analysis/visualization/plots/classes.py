"""Every drawn tile of the evaluation set on one map, coloured by its class."""

from __future__ import annotations

from collections.abc import Sequence

import ipywidgets as widgets
from matplotlib.lines import Line2D

from common.analysis.selector.models.selection import Selection
from common.analysis.visualization.common import mosaic, panels
from common.analysis.visualization.dataset.plots import tiles
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
            held = [boxes[one.tile] for one in drawn if one.label == label]
            axis.scatter(
                [(one.west_lon + one.east_lon) / 2.0 for one in held],
                [(one.min_lat + one.max_lat) / 2.0 for one in held],
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
