"""Every tile of Mars on one map: the ones the filter kept, and the ones it did not."""

from __future__ import annotations

from collections.abc import Sequence

import ipywidgets as widgets
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from analysis import configs
from analysis.selector.models.selection import Selection
from analysis.visualization.common import mosaic, panels
from analysis.visualization.common.models.box import Box
from shared.maths import tessellate

MAP_FIGURE_SIZE = (14.0, 7.6)
MARS = Box(-180.0, -90.0, 180.0, 90.0)
BASEMAP_PIXELS = 2400
RASTER_DEG = 0.1

KEPT = "#2ca02c"
EXCLUDED = "#d62728"
TILE_ALPHA = 0.45


def plot(picked: Sequence[Selection]) -> widgets.Widget:
    """Map every tile of Mars, the kept ones in green and every other one in red."""
    tile_km = configs.load().tile_km
    offsets = tessellate.tile_offsets(tile_km)
    kept = np.zeros(sum(tessellate.band_columns(tile_km)), dtype=bool)
    for one in picked:
        if one.tile.kept:
            kept[offsets[one.tile.band] + one.tile.column] = True
    return mosaic.fetched(
        MARS, lambda image: figure(kept, tile_km, image), BASEMAP_PIXELS
    )


def figure(kept: np.ndarray, tile_km: float, image: bytes) -> widgets.Widget:
    """Draw the mosaic of Mars with every tile coloured by what the filter kept."""
    # Every tile is painted where a fine lon/lat raster samples it, north row first
    lat = np.arange(MARS.north - RASTER_DEG / 2.0, MARS.south, -RASTER_DEG)
    lon = np.arange(MARS.west + RASTER_DEG / 2.0, MARS.east, RASTER_DEG)
    bands, columns = tessellate.tile_indices(
        *np.meshgrid(lat, lon, indexing="ij"), tile_km
    )
    held = kept[tessellate.tile_offsets(tile_km)[bands] + columns]
    painted = np.where(
        held[..., None],
        to_rgba(KEPT, TILE_ALPHA),
        to_rgba(EXCLUDED, TILE_ALPHA),
    )
    drawn, axis = panels.board(MAP_FIGURE_SIZE)
    mosaic.draw(axis, MARS, image)
    axis.imshow(painted, extent=MARS.extent, origin="upper", interpolation="nearest")
    axis.set_title(
        f"{int(kept.sum()):,} of {kept.size:,} tiles kept", fontsize=12, loc="left"
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.tick_params(labelsize=8)
    axis.legend(
        handles=[
            Patch(color=KEPT, alpha=TILE_ALPHA, label="kept"),
            Patch(color=EXCLUDED, alpha=TILE_ALPHA, label="excluded"),
        ],
        fontsize=9,
        loc="lower left",
    )
    drawn.tight_layout()
    return panels.rendered(drawn)
