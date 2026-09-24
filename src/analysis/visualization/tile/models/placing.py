"""Where a tile's ground falls back onto lon and lat."""

from __future__ import annotations

import numpy as np

from analysis.coverage.projection import footprints
from common.maths import geodesy
from common.maths.box import Crop, crop_around
from common.models.tile import Tile

MIN_SPAN_DEG = 0.5
RING_SAMPLES = 17


class Placed:
    """Where one tile's ground falls on the mosaic, in lon and lat."""

    def __init__(self, tile: Tile) -> None:
        """Project one tile and lay its bounds back onto lon and lat.

        Args:
            tile: The tile to place.
        """
        self._centre = (tile.centre_lon, tile.centre_lat)
        self._bounds = footprints.tile_shape(tile).bounds

    def outline(self) -> tuple[np.ndarray, np.ndarray]:
        """Trace the whole tile as a closed lon/lat ring.

        Returns:
            longitudes: The longitudes of the ring, closing where it opened.
            latitudes: The latitudes of the ring, closing where it opened.
        """
        west, south, east, north = self._bounds
        along = np.linspace(west, east, RING_SAMPLES)
        up = np.linspace(south, north, RING_SAMPLES)
        flat = np.full(RING_SAMPLES, 0.0)
        lon, lat = geodesy.laea_inverse(
            np.concatenate([along, flat + east, along[::-1], flat + west]),
            np.concatenate([flat + south, up, flat + north, up[::-1]]),
            *self._centre,
        )
        return self.around(lon), lat

    def around(self, lon: np.ndarray) -> np.ndarray:
        """Bring longitudes onto the same turn as the tile's own.

        Args:
            lon: The longitudes to bring around, in degrees.

        Returns:
            longitudes: The same longitudes, on the tile's own turn.
        """
        return self._centre[0] + geodesy.normalise_longitude(lon - self._centre[0])

    def box(self) -> Crop:
        """Return the lon/lat crop the whole tile falls in, held open to a minimum."""
        return crop_around(*self.outline(), MIN_SPAN_DEG)
