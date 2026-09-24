"""Where a tile's ground falls back onto lon and lat, and laying it on the mosaic."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis.coverage.projection import footprints
from analysis.utils.tile_group import tile_grid
from analysis.visualization import mosaic
from common.maths import box, geodesy
from common.maths.box import Crop, crop_around
from common.maths.geodesy import HALF_TURN
from common.models.tile import Tile

MIN_SPAN_DEG = 0.5
RING_SAMPLES = 17

TILE_EDGE = "#ffffff"
TILE_WIDTH = 1.4


class Placed:
    """Where one tile's ground falls on the mosaic, in lon and lat.

    Attributes:
        centre_lon: The longitude the tile is centred on, which sets its turn.
        lon: The longitudes of its closed outline, on its own turn.
        lat: The latitudes of its closed outline.
    """

    def __init__(self, tile: Tile) -> None:
        """Project one tile and trace its bounds back onto lon and lat.

        Args:
            tile: The tile to place.
        """
        self.centre_lon = tile.centre_lon
        west, south, east, north = footprints.tile_shape(tile).bounds
        along = np.linspace(west, east, RING_SAMPLES)
        up = np.linspace(south, north, RING_SAMPLES)
        flat = np.full(RING_SAMPLES, 0.0)
        lon, self.lat = geodesy.laea_inverse(
            np.concatenate([along, flat + east, along[::-1], flat + west]),
            np.concatenate([flat + south, up, flat + north, up[::-1]]),
            tile.centre_lon,
            tile.centre_lat,
        )
        self.lon = self.around(lon)

    def around(self, lon: np.ndarray) -> np.ndarray:
        """Bring longitudes onto the same turn as the tile's own.

        Args:
            lon: The longitudes to bring around, in degrees.

        Returns:
            longitudes: The same longitudes, on the tile's own turn.
        """
        return self.centre_lon + geodesy.normalise_longitude(lon - self.centre_lon)

    def box(self) -> Crop:
        """Return the lon/lat crop the whole tile falls in, held open to a minimum."""
        return crop_around(self.lon, self.lat, MIN_SPAN_DEG)


def placed(name: str, cut=None) -> Placed | None:
    """Lay one tile back onto the mosaic.

    Args:
        name: The tile's name, such as "b123_c0456".
        cut: Anything bounded by two latitudes and two longitudes it is cut to
            instead of its own box, or None.

    Returns:
        placed: Where it falls in lon and lat, or None where no crop covers it.
    """
    tile = tile_grid().tile_named(name)
    placed_tile = Placed(tile if cut is None else box.recut(tile, cut))
    # A tile wrapping the pole has no lon/lat box a plate carree crop can cover
    spread = placed_tile.lon.max() - placed_tile.lon.min()
    return placed_tile if spread <= HALF_TURN else None


def outlined_board(
    placed_tile: Placed, size: tuple[float, float], crop: Crop, image: bytes, **style
) -> tuple[Figure, Axes]:
    """Open a figure with one mosaic crop drawn on it, and the tile outlined on top.

    Args:
        placed_tile: The tile to outline.
        size: The figure's size in inches.
        crop: The lon/lat box the crop covers.
        image: The crop, as the mosaic fetched it.
        style: Anything further the outline is drawn with.

    Returns:
        figure: The figure the crop is drawn on.
        axis: The crop itself, in lon and lat, with the tile outlined.
    """
    figure, axis = mosaic.board(size, crop, image)
    axis.plot(
        placed_tile.lon, placed_tile.lat, color=TILE_EDGE, linewidth=TILE_WIDTH, **style
    )
    return figure, axis
