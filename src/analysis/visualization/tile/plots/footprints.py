"""The tile on show, with the footprint of every observation it keeps."""

from __future__ import annotations

import ipywidgets as widgets
from matplotlib.lines import Line2D

from analysis.stats.models.tile import TileLooks
from analysis.stats.tile import read
from analysis.visualization.common import mosaic, panels
from analysis.visualization.common.models.box import Box
from analysis.visualization.common.models.colours import Colour
from analysis.visualization.common.models.coverage import Coverage
from analysis.visualization.tile.models.placing import Placed
from analysis.visualization.tile.plots import outlines, placing

MAP_FIGURE_SIZE = (9.0, 6.0)

TILE_EDGE = "#ffffff"
TILE_WIDTH = 1.4
TRACE_WIDTH = 1.2
TRACE_ALPHA = 0.85

NOTE_COLOUR = "#ffffff"
NOTE_SIZE = 11

_NOTHING = "No footprints available"


def plot(coverage: Coverage) -> widgets.Widget:
    """Show the tile with the footprint of every observation it keeps."""
    if not coverage:
        return panels.unavailable()
    summary = coverage[0].summary
    grid = placing.placed(summary.tile)
    if grid is None:
        return panels.unavailable(mosaic.BASEMAP_FAILED.format(reason=mosaic.NO_BOX))
    looks = read.read_tile(coverage)
    box = grid.box()
    title = panels.title(coverage)
    return mosaic.fetched(
        box, lambda image: figure(grid, coverage, looks, box, image, title)
    )


def figure(
    grid: Placed,
    coverage: Coverage,
    looks: TileLooks | None,
    box: Box,
    image: bytes,
    title: str,
) -> widgets.Widget:
    """Draw the tile's crop with the footprints its window keeps traced on it."""
    drawn, axis = panels.board(MAP_FIGURE_SIZE)
    mosaic.draw(axis, box, image)
    lon, lat = grid.outline()
    axis.plot(lon, lat, color=TILE_EDGE, linewidth=TILE_WIDTH)
    traced: dict[str, Colour] = {}
    if looks is not None and looks.window.kept:
        track = looks.track
        shapes = outlines.read(coverage)
        colours = panels.colours(track.labels)
        for index in looks.taken:
            shape = shapes[track.observations[index].pdsid]
            label = track.labels[track.owners[index]]
            for line_lon, line_lat in outlines.traced(shape):
                axis.plot(
                    grid.around(line_lon),
                    line_lat,
                    color=colours[label],
                    linewidth=TRACE_WIDTH,
                    alpha=TRACE_ALPHA,
                )
                traced[label] = colours[label]
    if not traced:
        panels.note(axis, _NOTHING, colour=NOTE_COLOUR, size=NOTE_SIZE)
    axis.set_title(title, fontsize=12, loc="left")
    panels.key_beside(
        drawn,
        [
            Line2D([], [], color=colour, linewidth=TRACE_WIDTH, label=label)
            for label, colour in traced.items()
        ],
    )
    return panels.rendered(drawn)
