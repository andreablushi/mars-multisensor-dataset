"""Laying one tile back onto the mosaic."""

from __future__ import annotations

from dataclasses import replace

from analysis import configs
from analysis.visualization.tile.models.placing import Placed
from common.maths.geodesy import HALF_TURN
from common.maths.tessellate import Tessellate


def placed(name: str, cut=None) -> Placed | None:
    """Lay one tile back onto the mosaic.

    Args:
        name: The tile's name, such as "b123_c0456".
        cut: Anything bounded by two latitudes and two longitudes it is cut to
            instead of its own box, or None.

    Returns:
        placed: Where it falls in lon and lat, or None where no crop covers it.
    """
    tile = Tessellate.of(configs.load().tile_km).tile_named(name)
    if cut is not None:
        tile = replace(
            tile,
            min_lat=cut.min_lat,
            max_lat=cut.max_lat,
            west_lon=cut.west_lon,
            east_lon=cut.east_lon,
        )
    grid = Placed(tile)
    # A tile wrapping the pole has no lon/lat box a plate carree crop can cover
    lon, _ = grid.outline()
    return grid if lon.max() - lon.min() <= HALF_TURN else None
