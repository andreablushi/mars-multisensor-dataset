"""Where a tile's ground falls back onto lon and lat, and its outline on the mosaic."""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis.coverage.projection import footprints
from analysis.utils.tile_group import tile_grid
from common.maths import box, geodesy
from common.models.tile import Tile
from visualization import config, panels
from visualization.mosaic import read_mosaic


class PlacedTile:
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
        west, south, east, north = footprints.laea_tile(tile).bounds
        along = np.linspace(west, east, config.RING_SAMPLES)
        up = np.linspace(south, north, config.RING_SAMPLES)
        flat = np.full(config.RING_SAMPLES, 0.0)
        lon, self.lat = geodesy.laea_inverse(
            np.concatenate([along, flat + east, along[::-1], flat + west]),
            np.concatenate([flat + south, up, flat + north, up[::-1]]),
            tile.centre_lon,
            tile.centre_lat,
        )
        self.lon = self.around(lon)

    def around(self, lon: np.ndarray) -> np.ndarray:
        """Bring longitudes onto the same turn as the tile's own."""
        return self.centre_lon + geodesy.normalise_longitude(lon - self.centre_lon)

    def crop(self) -> box.Crop:
        """Return the lon/lat crop the whole tile falls in, held open to a minimum."""
        return box.crop_around(self.lon, self.lat, config.MIN_SPAN_DEG)


def placed_tile(name: str, cut=None) -> PlacedTile | None:
    """Lay one tile back onto the mosaic.

    Args:
        name: The tile's name, such as "b123_c0456".
        cut: Anything bounded by two latitudes and two longitudes it is cut to
            instead of its own box, or None.

    Returns:
        placed: Where it falls in lon and lat, or None where no crop covers it.
    """
    tile = tile_grid().tile_named(name)
    placed = PlacedTile(tile if cut is None else box.recut(tile, cut))
    # A tile wrapping the pole has no lon/lat box a plate carree crop can cover
    return placed if placed.lon.max() - placed.lon.min() <= geodesy.HALF_TURN else None


def outlined_board(
    placed: PlacedTile, size: tuple[float, float], image: bytes, **style
) -> tuple[Figure, Axes]:
    """Open a figure with the tile's mosaic crop drawn on it, and the tile outlined.

    Args:
        placed: The tile to outline.
        size: The figure's size in inches.
        image: The tile's crop, as the mosaic fetched it.
        style: Anything further the outline is drawn with.

    Returns:
        figure: The figure the crop is drawn on.
        axis: The crop itself, in lon and lat, with the tile outlined.
    """
    crop = placed.crop()
    figure = Figure(figsize=size)
    axis = figure.subplots()
    axis.imshow(read_mosaic(image), extent=crop.extent, origin="upper", cmap="gray")
    axis.set_aspect(1.0 / geodesy.longitude_stretch(crop.centre_lat))
    axis.set_xlim(crop.west, crop.east)
    axis.set_ylim(crop.south, crop.north)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.tick_params(labelsize=8)
    # A footprint reaching well past the crop is cut to it rather than framed
    axis.autoscale(False)
    axis.plot(
        placed.lon,
        placed.lat,
        color=config.TILE_EDGE,
        linewidth=config.TILE_WIDTH,
        **style,
    )
    return figure, axis


def tile_map(placed: PlacedTile, title: str, image: bytes) -> widgets.Image:
    """Draw the tile's crop of the mosaic, with the ground it covers outlined.

    Args:
        placed: Where the tile falls in lon and lat.
        title: What the map is titled.
        image: The tile's crop, as the mosaic fetched it.

    Returns:
        map: The map, rendered.
    """
    drawn, axis = outlined_board(placed, (7.0, 6.0), image, zorder=3)
    panels.titled(axis, title)
    drawn.tight_layout()
    return panels.rendered(drawn)
