"""What a MOLA gridded tile is published as, and where it lands."""

from __future__ import annotations

import re

from building.common.layout import GROUND, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache
from utils.disk import paths

# The one plane of a tile that is read, the height of its ground.
TOPOGRAPHY = "topography"
KINDS = (TOPOGRAPHY,)

# How a plane is spelled, named for its corner and step. Its kind drops polar tiles.
NAMING = Naming(
    re.compile(r"^(?:meg(?P<marker>[tc]))?(?P<tile>\d{2}[ns]\d{3}(?P<step>[cefgh])b)$"),
    identity="{tile}",
    marks=("marker",),
    template="meg{marker}{tile}",
    fields={TOPOGRAPHY: {"marker": "t"}},
)

# What the arrays of one tile hold, and which of them is stored for.
LAYOUT = Layout(
    instrument="MOLA",
    dims=("line", "sample"),
    axes=(GROUND, GROUND),
    measurement="topography",
)

# Where a tile is kept, in the one directory of the tile.
CACHE = ProductCache(paths.MOLA_ROOT, {None: (".lbl", ".img")})

# How fine a grid each resolution letter stands for, in pixels per degree.
RESOLUTIONS = {"c": 4, "e": 16, "f": 32, "g": 64, "h": 128}

# The grid a feature is mosaicked on, named for the record and how fine it is.
CYLINDRICAL = "megdr128"
GRIDS = {CYLINDRICAL: 128}


def resolution(tile: str) -> int:
    """Read how fine a grid one tile is written on.

    Args:
        tile: The tile, such as 00n180hb.

    Returns:
        The pixels per degree the tile holds.

    Raises:
        ValueError: When the tile is not one this can read.
    """
    parts = NAMING.parts(tile)
    if not parts:
        raise ValueError(f"{tile} is not a simple cylindrical MEGDR tile.")
    return RESOLUTIONS[parts["step"]]
