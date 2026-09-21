"""Laying one tile back onto the mosaic."""

from __future__ import annotations

from analysis import configs
from analysis.visualization.tile.models.placing import Placed
from common.maths.geodesy import HALF_TURN
from common.maths.tessellate import Tessellate


def placed(name: str) -> Placed | None:
    """Lay one tile back onto the mosaic.

    Args:
        name: The tile's name, such as "b123_c0456".

    Returns:
        placed: Where it falls in lon and lat, or None where no crop covers it.
    """
    grid = Placed(Tessellate.of(configs.load().tile_km).tile_named(name))
    # A tile wrapping the pole has no lon/lat box a plate carree crop can cover
    lon, _ = grid.outline()
    return grid if lon.max() - lon.min() <= HALF_TURN else None
