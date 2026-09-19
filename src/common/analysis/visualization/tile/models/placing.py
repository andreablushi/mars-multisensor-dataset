"""Where a tile's ground falls back onto lon and lat."""

from __future__ import annotations

import numpy as np

from common.analysis.coverage.projection.geometry import footprints
from common.analysis.visualization.common.models.box import Box
from common.maths import geodesy
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
        region = footprints.tile_region(tile)
        self._centre = (region.centre_lon, region.centre_lat)
        self._bounds = region.shape.bounds

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

    def box(self) -> Box:
        """Return the lon/lat box the whole tile falls in, held open to a minimum.

        Returns:
            box: The box a mosaic crop is asked for over.
        """
        lon, lat = self.outline()
        centre_lat = float((lat.min() + lat.max()) / 2.0)
        south, north = _floored(float(lat.min()), float(lat.max()), MIN_SPAN_DEG)
        west, east = _floored(
            float(lon.min()),
            float(lon.max()),
            MIN_SPAN_DEG / geodesy.longitude_stretch(centre_lat),
        )
        return Box(west, south, east, north)


def _floored(low: float, high: float, minimum: float) -> tuple[float, float]:
    """Hold a side of a box open to a minimum width, about its middle.

    Args:
        low: The lower edge.
        high: The upper edge.
        minimum: The width to hold it open to.

    Returns:
        edges: The edges, widened about their middle when they sit closer than the
            minimum.
    """
    if high - low >= minimum:
        return low, high
    centre = (low + high) / 2.0
    return centre - minimum / 2.0, centre + minimum / 2.0
