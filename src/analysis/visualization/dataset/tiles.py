"""Every tile of Mars on one map: the ones the filter kept, and the ones it did not."""

from __future__ import annotations

from collections.abc import Sequence

import ipywidgets as widgets
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from analysis.selector.models.selection import Selection
from analysis.utils.tile_group import tile_grid
from analysis.visualization import mosaic, panels
from analysis.visualization.mosaic import MARS
from common.maths.tessellate import Tessellate

RASTER_DEG = 0.1

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
        MARS, lambda image: kept_map(kept, grid, image), mosaic.MARS_PIXELS
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
    title = f"{int(kept.sum()):,} of {kept.size:,} tiles kept"
    figure, axis = mosaic.mars_board(image, title)
    axis.imshow(painted, **mosaic.MARS_LAID)
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
