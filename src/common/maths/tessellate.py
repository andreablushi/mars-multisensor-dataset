"""Splitting Mars into equal-area tiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from common.maths.geodesy import HALF_TURN, TURN
from common.maths.physics import RADIUS_KM
from common.models.tile import Tile


@lru_cache(maxsize=4)
def split_bands(tile_km: float) -> tuple[float, ...]:
    """Return the latitudes the grid's bands are bounded by.

    Args:
        tile_km: The side a tile is sized to, in kilometres.

    Returns:
        edges: One latitude per boundary, from the south pole north, holding one
            more than there are bands.
    """
    bands = max(1, round(math.pi * RADIUS_KM / tile_km))
    # The latitudinal boundaries, of equal height from pole to pole.
    return tuple(float(edge) for edge in np.linspace(-90.0, 90.0, bands + 1))


@lru_cache(maxsize=4)
def split_bands_columns(tile_km: float) -> tuple[int, ...]:
    """Return how many columns each latitude band is split into.

    Args:
        tile_km: The side a tile is sized to, in kilometres.

    Returns:
        columns: One count per band, from the south pole north, each band holding
            as many tiles as its area has room for.
    """
    # The band area, a zone of the sphere and so its height taken in sine.
    areas = (
        2.0 * math.pi * RADIUS_KM**2 * np.diff(np.sin(np.radians(split_bands(tile_km))))
    )
    return tuple(max(1, round(area / tile_km**2)) for area in areas)


@dataclass(frozen=True, slots=True, eq=False)
class Tessellate:
    """The grid one tile size splits Mars into, and every query put to it.

    Attributes:
        edges: The latitude each band is bounded by, from the south pole north.
        columns: How many columns each band is split into.
        offsets: Where each band's first tile stands when every tile is counted
            in one run.
    """

    edges: tuple[float, ...]
    columns: tuple[int, ...]
    offsets: np.ndarray

    @classmethod
    def of(cls, tile_km: float) -> Tessellate:
        """Return the grid one tile size splits Mars into.

        Args:
            tile_km: The side a tile is sized to, in kilometres.

        Returns:
            grid: The grid, its bands counted once for every query put to it.
        """
        columns = split_bands_columns(tile_km)
        return cls(
            split_bands(tile_km),
            columns,
            np.concatenate(([0], np.cumsum(columns)[:-1])),
        )

    def tile_of(self, band: int, column: int) -> Tile:
        """Return one tile from where it sits on the grid.

        Args:
            band: The latitude band, counted from the south pole.
            column: The place along that band, counted east from the prime meridian.

        Returns:
            tile: The tile, a single cap circling the pole where its band holds one.
        """
        held = self.columns[band]
        return Tile(
            band=band,
            column=column,
            min_lat=self.edges[band],
            max_lat=self.edges[band + 1],
            west_lon=0.0 if held == 1 else TURN * column / held,
            east_lon=0.0 if held == 1 else TURN * (column + 1) / held,
        )

    def tile_named(self, name: str) -> Tile:
        """Return the tile a name spells.

        Args:
            name: The tile's name, such as "b123_c0456".

        Returns:
            tile: The tile it names.
        """
        band, column = name.split("_")
        return self.tile_of(int(band[1:]), int(column[1:]))

    def tile_indices(
        self, lat: np.ndarray | float, lon: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the tile every point falls in.

        Args:
            lat: The latitudes in degrees.
            lon: The longitudes in degrees, any turn.

        Returns:
            bands: The band each point falls in.
            columns: The column each point falls in along its band.
        """
        counts = np.asarray(self.columns)
        bands = np.clip(
            np.floor((np.asarray(lat, dtype=float) + 90.0) / HALF_TURN * counts.size),
            0,
            counts.size - 1,
        ).astype(np.int64)
        held = counts[bands]
        along = np.floor(np.mod(np.asarray(lon, dtype=float), TURN) / TURN * held)
        return bands, np.minimum(along.astype(np.int64), held - 1)

    def flat_tile_indices(
        self, band: np.ndarray | int, column: np.ndarray | int
    ) -> np.ndarray:
        """Return where each tile stands when every tile is counted in one run.

        Args:
            band: The latitude band of each tile, one or an array of them.
            column: The place of each along its band, one or an array of them.

        Returns:
            indices: The place of each tile among all of them, south to north and
                west to east.
        """
        return self.offsets[band] + np.asarray(column, dtype=np.int64)
