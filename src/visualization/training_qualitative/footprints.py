"""The tile on show, with the footprint of every observation it keeps."""

from __future__ import annotations

from functools import lru_cache, partial

import ipywidgets as widgets
import numpy as np
from matplotlib.lines import Line2D
from shapely import wkt
from shapely.geometry.base import BaseGeometry

from analysis import paths
from analysis.metadata.loaders.observations import load_observations
from analysis.models.instrument import InstrumentSet
from analysis.stats.models import TileTrack
from analysis.stats.tile import read_tile_track
from analysis.utils.tile_group import group_of_tile_named
from visualization import mosaic, panels
from visualization.config import NO_BOX
from visualization.panels import Colour
from visualization.placement import PlacedTile, outlined_board, placed_tile
from visualization.training_qualitative import config
from visualization.training_qualitative.charts import note
from visualization.training_qualitative.picker import Coverage, tile_title


def plot(coverage: Coverage) -> widgets.Widget:
    """Show the tile with the footprint of every observation it keeps."""
    placed = placed_tile(coverage[0].summary.tile)
    if placed is None:
        return panels.unavailable(NO_BOX)
    tile_track = read_tile_track(coverage)
    return mosaic.fetched(
        placed.crop(), partial(footprints_map, placed, coverage, tile_track)
    )


def footprints_map(
    placed: PlacedTile, coverage: Coverage, tile_track: TileTrack | None, image: bytes
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
        group = group_of_tile_named(coverage[0].summary.tile)
        footprints = {
            pdsid: footprint
            for instrument in coverage
            if instrument.observed
            for pdsid, footprint in set_footprints(
                group, instrument.summary.set_key
            ).items()
        }
        colours = panels.colours(track.labels)
        for index in tile_track.taken:
            footprint = footprints[track.observations[index].pdsid]
            label = track.labels[track.owners[index]]
            for part in getattr(footprint, "geoms", [footprint]):
                ring = np.asarray(getattr(part, "exterior", part).coords, dtype=float)
                axis.plot(
                    placed.around(ring[:, 0]),
                    ring[:, 1],
                    color=colours[label],
                    linewidth=config.TRACE_WIDTH,
                    alpha=0.85,
                )
                traced_colours[label] = colours[label]
    if not traced_colours:
        note(axis, "No footprints available", colour="#ffffff", size=11)
    panels.titled(axis, tile_title(coverage))
    panels.key_beside(
        drawn,
        [
            Line2D([], [], color=colour, linewidth=config.TRACE_WIDTH, label=label)
            for label, colour in traced_colours.items()
        ],
    )
    return panels.rendered(drawn)


@lru_cache(maxsize=4)
def set_footprints(group: str, set_key: str) -> dict[str, BaseGeometry]:
    """Read one instrument set's published footprints, held for its panels.

    Args:
        group: The tile group the set was downloaded for.
        set_key: The set's key.

    Returns:
        footprints: Its published footprints, by pdsid.
    """
    path = paths.metadata_path(group, InstrumentSet.from_key(set_key))
    return {
        observation.pdsid: wkt.loads(observation.wkt)
        for observation in load_observations(path).observations
    }
