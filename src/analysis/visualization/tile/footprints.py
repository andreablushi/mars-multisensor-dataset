"""The tile on show, with the footprint of every observation it keeps."""

from __future__ import annotations

import ipywidgets as widgets
from matplotlib.lines import Line2D

from analysis.stats.models import TileTrack
from analysis.stats.tile import read_tile_track
from analysis.visualization import mosaic, panels
from analysis.visualization.panels import Colour, Coverage
from analysis.visualization.tile import outlines
from analysis.visualization.tile.placement import (
    PlacedTile,
    outlined_board,
    placed_tile,
)

TRACE_WIDTH = 1.2


def plot(coverage: Coverage) -> widgets.Widget:
    """Show the tile with the footprint of every observation it keeps."""
    placed = placed_tile(coverage[0].summary.tile)
    if placed is None:
        return panels.unavailable(mosaic.NO_BOX)
    tile_track = read_tile_track(coverage)
    return mosaic.fetched(
        placed.box(),
        lambda image: footprints_map(placed, coverage, tile_track, image),
    )


def footprints_map(
    placed: PlacedTile,
    coverage: Coverage,
    tile_track: TileTrack | None,
    image: bytes,
) -> widgets.Image:
    """Draw the tile's crop with the footprints its window keeps traced on it.

    Args:
        placed: Where the tile falls in lon and lat.
        coverage: The tile's instrument sets, whose footprints are read.
        tile_track: Its track and the observations it keeps, or None.
        image: The tile's crop, as the mosaic fetched it.

    Returns:
        map: The map, rendered.
    """
    drawn, axis = outlined_board(placed, (9.0, 6.0), image)
    traced_colours: dict[str, Colour] = {}
    if tile_track is not None and tile_track.window.kept:
        track = tile_track.track
        footprints = outlines.read_footprints(coverage)
        colours = panels.colours(track.labels)
        for index in tile_track.taken:
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
