"""What a MOLA gridded sheet is published as, and where it lands."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from building import paths
from building.common.layout import Axis, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache


class Kind(StrEnum):
    """The one plane of a sheet that is read, the height of its ground."""

    TOPOGRAPHY = "topography"


# How a plane is spelled, named for its corner and step. Its kind drops polar sheets.
NAMING = Naming(
    re.compile(
        r"^(?:meg(?P<marker>[tc]))?(?P<sheet>\d{2}[ns]\d{3}(?P<step>[cefgh])b)$"
    ),
    identity="{sheet}",
    marks=("marker",),
    template="meg{marker}{sheet}",
    fields={Kind.TOPOGRAPHY: {"marker": "t"}},
)

# What the arrays of one sheet hold, and which of them is stored for.
LAYOUT = Layout(
    instrument="MOLA",
    dims=("line", "sample"),
    axes=(Axis.GROUND, Axis.GROUND),
    measurement="elevation",
    beside={"delay": ("line", "sample"), "delay_inside": ("line", "sample")},
    stored="int16",
)

# Where a sheet is kept, in the one directory of the sheet.
CACHE = ProductCache(paths.MOLA_ROOT, NAMING, {None: (".lbl", ".img")})

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
        resolution: How many bins of it one degree holds.
        product: The single product a polar cap is published as, or None for sheets.
    """

    resolution: int
    product: str | None = None


# The grid a tile is merged from, named for the record and how fine it is.
EQUATORIAL = "megdr128"
COARSE = "megdr64"
NORTH_POLAR = "megdr128n"
SOUTH_POLAR = "megdr128s"

GRIDS = {
    EQUATORIAL: Grid(128),
    COARSE: Grid(64),
    NORTH_POLAR: Grid(128, "megt_n_128_1"),
    SOUTH_POLAR: Grid(128, "megt_s_128_1"),
}
