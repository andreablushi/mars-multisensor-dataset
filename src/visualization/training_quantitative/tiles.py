"""Every tile of Mars on one map: the ones the filter kept, and the ones it did not."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import ipywidgets as widgets
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from analysis.partition import Tessellate
from analysis.selector.models.selection import Selection
from analysis.utils.tile_group import tile_grid
from visualization import mosaic, panels
from visualization.config import MARS_PIXELS
from visualization.mosaic import MARS
from visualization.training_quantitative import config


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
    return mosaic.fetched(MARS, partial(kept_map, kept, grid), MARS_PIXELS)


def kept_map(kept: np.ndarray, grid: Tessellate, image: bytes) -> widgets.Image:
    """Draw the mosaic of Mars with every tile coloured by what the filter kept.

    Args:
        kept: Whether the filter kept each tile, by flat tile index.
        grid: The tiling Mars is split into.
        image: The mosaic of the whole planet, as fetched.

    Returns:
        map: The map, rendered.
    """
    lat = np.arange(
        MARS.north - config.RASTER_DEG / 2.0, MARS.south, -config.RASTER_DEG
    )
    lon = np.arange(MARS.west + config.RASTER_DEG / 2.0, MARS.east, config.RASTER_DEG)
    bands, columns = grid.tile_indices(*np.meshgrid(lon, lat))
    painted = np.where(
        kept[grid.flat_tile_indices(bands, columns)][..., None],
        to_rgba(config.KEPT, config.TILE_ALPHA),
        to_rgba(config.EXCLUDED, config.TILE_ALPHA),
    )
    title = f"{int(kept.sum()):,} of {kept.size:,} tiles kept"
    figure, axis = mosaic.mars_board(image, title)
    axis.imshow(painted, **mosaic.MARS_LAID)
    axis.legend(
        handles=[
            Patch(color=config.KEPT, alpha=config.TILE_ALPHA, label="kept"),
            Patch(color=config.EXCLUDED, alpha=config.TILE_ALPHA, label="excluded"),
        ],
        fontsize=9,
        loc="lower left",
    )
    figure.tight_layout()
    return panels.rendered(figure)
