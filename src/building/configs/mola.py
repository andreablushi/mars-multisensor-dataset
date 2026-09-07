"""What a MOLA gridded tile is published as, and where it lands."""

from __future__ import annotations

import re
from dataclasses import dataclass

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

# The latitude the tiled grid reaches, past which only a cap is published fine.
TILED_REACH = 88.0

# The latitude a cap holds at every longitude, its corners alone reaching lower.
CAP_FLOOR = 51.55


@dataclass(frozen=True, slots=True)
class Grid:
    """One grid of the gridded record, and what it is published as.

    Attributes:
        name: What the grid is called, which is also what every crop merged
            from it is stored under.
        resolution: How many bins of it one degree holds.
        product: The single product it is published as, and None for a grid
            published as the tiles that cover it.
        north: Whether it is centred on the north pole, and None where it is
            cylindrical and centred on no pole at all.
    """

    name: str
    resolution: int
    product: str | None = None
    north: bool | None = None

    @property
    def polar(self) -> bool:
        """Say whether the grid is projected onto a pole.

        Returns:
            True where it is a cap, and False where it is cylindrical.
        """
        return self.north is not None


# The grid a feature is merged from, named for the record and how fine it is.
CYLINDRICAL = "megdr128"
COARSE = "megdr64"
NORTH_CAP = "megdr128n"
SOUTH_CAP = "megdr128s"

GRIDS = {
    CYLINDRICAL: Grid(CYLINDRICAL, 128),
    COARSE: Grid(COARSE, 64),
    NORTH_CAP: Grid(NORTH_CAP, 128, "megt_n_128_1", north=True),
    SOUTH_CAP: Grid(SOUTH_CAP, 128, "megt_s_128_1", north=False),
}


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
