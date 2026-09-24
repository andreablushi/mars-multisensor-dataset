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

from analysis.selector.models.selection import Selection
from analysis.utils.tile_group import tile_grid
from analysis.visualization import mosaic, panels
from common.maths.box import Crop
from common.maths.physics import RADIUS_M
from common.maths.tessellate import Tessellate

MAP_FIGURE_SIZE = (14.0, 7.6)
MARS = Crop(-180.0, -90.0, 180.0, 90.0)
BASEMAP_PIXELS = 2400
RASTER_DEG = 0.1

GLOBE = crs.Globe(semimajor_axis=RADIUS_M, semiminor_axis=RADIUS_M, ellipse=None)
LONLAT = crs.PlateCarree(globe=GLOBE)
ROBINSON = crs.Robinson(globe=GLOBE)
GRATICULE = "#ffffff"

LAID = dict(
    extent=MARS.extent,
    origin="upper",
    interpolation="nearest",
    transform=LONLAT,
    regrid_shape=(BASEMAP_PIXELS, BASEMAP_PIXELS // 2),
)

KEPT = "#2ca02c"
EXCLUDED = "#d62728"
TILE_ALPHA = 0.45


def plot(selections: Sequence[Selection]) -> widgets.Widget:
    """Map every tile of Mars, the kept ones in green and every other one in red."""
    grid = tile_grid()
    kept = np.zeros(sum(grid.columns), dtype=bool)
    kept_tiles = [selection.tile for selection in selections if selection.tile.kept]
    kept[
        grid.flat_tile_indices(
            [tile.band for tile in kept_tiles], [tile.column for tile in kept_tiles]
        )
    ] = True
    return mosaic.fetched(
        MARS, lambda image: kept_map(kept, grid, image), BASEMAP_PIXELS
    )


def kept_map(kept: np.ndarray, grid: Tessellate, image: bytes) -> widgets.Image:
    """Draw the mosaic of Mars with every tile coloured by what the filter kept.

    Args:
        kept: Whether the filter kept each tile, by flat tile index.
        grid: The tiling Mars is split into.
        image: The mosaic of the whole planet, as fetched.

    Returns:
        map: The map, rendered.
    """
    lat = np.arange(MARS.north - RASTER_DEG / 2.0, MARS.south, -RASTER_DEG)
    lon = np.arange(MARS.west + RASTER_DEG / 2.0, MARS.east, RASTER_DEG)
    bands, columns = grid.tile_indices(*np.meshgrid(lat, lon, indexing="ij"))
    painted = np.where(
        kept[grid.flat_tile_indices(bands, columns)][..., None],
        to_rgba(KEPT, TILE_ALPHA),
        to_rgba(EXCLUDED, TILE_ALPHA),
    )
    figure, axis = mars_board(image, f"{int(kept.sum()):,} of {kept.size:,} tiles kept")
    axis.imshow(painted, **LAID)
    axis.legend(
        handles=[
            Patch(color=KEPT, alpha=TILE_ALPHA, label="kept"),
            Patch(color=EXCLUDED, alpha=TILE_ALPHA, label="excluded"),
        ],
        fontsize=9,
        loc="lower left",
    )
    figure.tight_layout()
    return panels.rendered(figure)


def mars_board(image: bytes, title: str) -> tuple[Figure, Axes]:
    """Draw the mosaic of Mars under its graticule, for tiles to be marked on.

    Args:
        image: The mosaic of the whole planet, as fetched.
        title: What the map is titled.

    Returns:
        figure: The figure the map is drawn on.
        axis: The map itself, in lon and lat.
    """
    figure, axis = panels.board(MAP_FIGURE_SIZE, ROBINSON)
    axis.set_global()
    axis.imshow(mosaic.read_mosaic(image), cmap="gray", **LAID)
    lines = axis.gridlines(
        LONLAT, draw_labels=True, color=GRATICULE, linewidth=0.4, alpha=0.5
    )
    lines.xlabel_style = lines.ylabel_style = {"size": 8}
    axis.set_title(title, fontsize=12, loc="left")
    return figure, axis
