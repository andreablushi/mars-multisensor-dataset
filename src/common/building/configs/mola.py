"""What a MOLA gridded sheet is published as, and where it lands."""

from __future__ import annotations

import re
from dataclasses import dataclass

from common.building import paths
from common.building.common.layout import GROUND, Layout
from common.building.common.naming import Naming
from common.building.common.product_cache import ProductCache

# The one plane of a sheet that is read, the height of its ground.
TOPOGRAPHY = "topography"
KINDS = (TOPOGRAPHY,)

# How a plane is spelled, named for its corner and step. Its kind drops polar sheets.
NAMING = Naming(
    re.compile(
        r"^(?:meg(?P<marker>[tc]))?(?P<sheet>\d{2}[ns]\d{3}(?P<step>[cefgh])b)$"
    ),
    identity="{sheet}",
    marks=("marker",),
    template="meg{marker}{sheet}",
    fields={TOPOGRAPHY: {"marker": "t"}},
)

# What the arrays of one sheet hold, and which of them is stored for.
LAYOUT = Layout(
    instrument="MOLA",
    dims=("line", "sample"),
    axes=(GROUND, GROUND),
    measurement="elevation",
    beside={"delay": ("line", "sample"), "delay_inside": ("line", "sample")},
    stored="int16",
)

# Where a sheet is kept, in the one directory of the sheet.
CACHE = ProductCache(paths.MOLA_ROOT, {None: (".lbl", ".img")})

# How fine a grid each resolution letter stands for, in pixels per degree.
RESOLUTIONS = {"c": 4, "e": 16, "f": 32, "g": 64, "h": 128}

# The latitude the sheeted grid reaches, past which only a cap is published fine.
SHEETED_REACH = 88.0

# The latitude a cap holds at every longitude, its corners alone reaching lower.
POLAR_FLOOR = 51.55


@dataclass(frozen=True, slots=True)
class Grid:
    """One grid of the gridded record, and what it is published as.

    Attributes:
        name: What the grid is called, which is also what every crop merged
            from it is stored under.
        resolution: How many bins of it one degree holds.
        product: The single product it is published as, and None for a grid
            published as the sheets that cover it.
        north: Whether it is centred on the north pole, and None where it is
            equatorial and centred on no pole at all.
    """

    name: str
    resolution: int
    product: str | None = None
    north: bool | None = None


# The grid a tile is merged from, named for the record and how fine it is.
EQUATORIAL = "megdr128"
COARSE = "megdr64"
NORTH_POLAR = "megdr128n"
SOUTH_POLAR = "megdr128s"

GRIDS = {
    EQUATORIAL: Grid(EQUATORIAL, 128),
    COARSE: Grid(COARSE, 64),
    NORTH_POLAR: Grid(NORTH_POLAR, 128, "megt_n_128_1", north=True),
    SOUTH_POLAR: Grid(SOUTH_POLAR, 128, "megt_s_128_1", north=False),
}
