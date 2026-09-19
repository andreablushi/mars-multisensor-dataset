"""Every tile of Mars on one map: the ones the filter kept, and the ones it did not."""

from __future__ import annotations

from collections.abc import Sequence

import ipywidgets as widgets
import numpy as np
from cartopy import crs
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from common.analysis import configs
from common.analysis.selector.models.selection import Selection
from common.analysis.visualization.common import mosaic, panels
from common.analysis.visualization.common.models.box import Box
from common.maths.physics import RADIUS_M
from common.maths.tessellate import Tessellate

MAP_FIGURE_SIZE = (14.0, 7.6)
MARS = Box(-180.0, -90.0, 180.0, 90.0)
BASEMAP_PIXELS = 2400
RASTER_DEG = 0.1
REGRID = (BASEMAP_PIXELS, BASEMAP_PIXELS // 2)

GLOBE = crs.Globe(semimajor_axis=RADIUS_M, semiminor_axis=RADIUS_M, ellipse=None)
LONLAT = crs.PlateCarree(globe=GLOBE)
ROBINSON = crs.Robinson(globe=GLOBE)
GRATICULE = "#ffffff"

LAID = dict(
    extent=MARS.extent,
    origin="upper",
    interpolation="nearest",
    transform=LONLAT,
    regrid_shape=REGRID,
)

KEPT = "#2ca02c"
EXCLUDED = "#d62728"
TILE_ALPHA = 0.45


def plot(picked: Sequence[Selection]) -> widgets.Widget:
    """Map every tile of Mars, the kept ones in green and every other one in red."""
    grid = Tessellate.of(configs.load().tile_km)
    kept = np.zeros(sum(grid.columns), dtype=bool)
    held = [one.tile for one in picked if one.tile.kept]
    kept[
        grid.flat_tile_indices([one.band for one in held], [one.column for one in held])
    ] = True
    return mosaic.fetched(MARS, lambda image: figure(kept, grid, image), BASEMAP_PIXELS)


def figure(kept: np.ndarray, grid: Tessellate, image: bytes) -> widgets.Widget:
    """Draw the mosaic of Mars with every tile coloured by what the filter kept."""
    lat = np.arange(MARS.north - RASTER_DEG / 2.0, MARS.south, -RASTER_DEG)
    lon = np.arange(MARS.west + RASTER_DEG / 2.0, MARS.east, RASTER_DEG)
    bands, columns = grid.tile_indices(*np.meshgrid(lat, lon, indexing="ij"))
    held = kept[grid.flat_tile_indices(bands, columns)]
    painted = np.where(
        held[..., None],
        to_rgba(KEPT, TILE_ALPHA),
        to_rgba(EXCLUDED, TILE_ALPHA),
    )
    drawn, axis = mars_board(image, f"{int(kept.sum()):,} of {kept.size:,} tiles kept")
    axis.imshow(painted, **LAID)
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


def mars_board(image: bytes, title: str) -> tuple[Figure, Axes]:
    """Draw the mosaic of Mars under its graticule, for tiles to be marked on.

    Args:
        image: The mosaic of the whole planet, as fetched.
        title: What the map is titled.

    Returns:
        figure: The figure the map is drawn on.
        axis: The map itself, in lon and lat.
    """
    drawn, axis = panels.board(MAP_FIGURE_SIZE, ROBINSON)
    axis.set_global()
    axis.imshow(mosaic.read_mosaic(image), cmap="gray", **LAID)
    lines = axis.gridlines(
        LONLAT, draw_labels=True, color=GRATICULE, linewidth=0.4, alpha=0.5
    )
    lines.xlabel_style = lines.ylabel_style = {"size": 8}
    axis.set_title(title, fontsize=12, loc="left")
    return drawn, axis
