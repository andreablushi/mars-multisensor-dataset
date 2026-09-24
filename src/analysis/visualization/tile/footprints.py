"""The tile on show, with the footprint of every observation it keeps."""

from __future__ import annotations

import ipywidgets as widgets
from matplotlib.lines import Line2D

from analysis.stats.models import TileLooks
from analysis.stats.tile import read_tile
from analysis.visualization import mosaic, panels
from analysis.visualization.panels import Colour, Coverage
from analysis.visualization.tile import outlines, placing
from analysis.visualization.tile.placing import Placed
from common.maths.box import Crop

TRACE_WIDTH = 1.2


def plot(coverage: Coverage) -> widgets.Widget:
    """Show the tile with the footprint of every observation it keeps."""
    if not coverage:
        return panels.unavailable()
    placed = placing.placed(coverage[0].summary.tile)
    if placed is None:
        return panels.unavailable(mosaic.NO_BOX)
    tile_looks = read_tile(coverage)
    box = placed.box()
    return mosaic.fetched(
        box, lambda image: figure(placed, coverage, tile_looks, box, image)
    )


def figure(
    placed: Placed,
    coverage: Coverage,
    tile_looks: TileLooks | None,
    box: Crop,
    image: bytes,
) -> widgets.Widget:
    """Draw the tile's crop with the footprints its window keeps traced on it."""
    drawn, axis = placing.outlined_board(placed, (9.0, 6.0), box, image)
    traced_colours: dict[str, Colour] = {}
    if tile_looks is not None and tile_looks.window.kept:
        track = tile_looks.track
        footprints = outlines.read_footprints(coverage)
        colours = panels.colours(track.labels)
        for index in tile_looks.taken:
            footprint = footprints[track.observations[index].pdsid]
            label = track.labels[track.owners[index]]
            for line_lon, line_lat in outlines.footprint_lines(footprint):
                axis.plot(
                    placed.around(line_lon),
                    line_lat,
                    color=colours[label],
                    linewidth=TRACE_WIDTH,
                    alpha=0.85,
                )
                traced_colours[label] = colours[label]
    if not traced_colours:
        panels.note(axis, "No footprints available", colour="#ffffff", size=11)
    axis.set_title(panels.title(coverage), fontsize=12, loc="left")
    panels.key_beside(
        drawn,
        [
            Line2D([], [], color=colour, linewidth=TRACE_WIDTH, label=label)
            for label, colour in traced_colours.items()
        ],
    )
    return panels.rendered(drawn)
