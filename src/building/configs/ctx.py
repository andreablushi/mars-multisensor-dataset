"""What a CTX scan is published as, where it lands, and how ISIS projects it."""

from __future__ import annotations

import re

from building import paths
from building.common.layout import GROUND, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache

IMAGE_SUFFIX = ".img"

CUBE_SUFFIX = ".cub"

# What the scan's own metadata is kept as, which ODE answers with rather than serves.
METADATA_SUFFIX = ".ode.json"

# What ODE says of a scan that its projected label does not, the geometry it was at
ODE_ACQUISITION = (
    "Incidence_angle",
    "Emission_angle",
    "Phase_angle",
    "Solar_longitude",
    "Solar_distance",
    "Solar_time",
)

PIXEL_RESOLUTION_M = 5.0

WARP_ALGORITHM = "forwardpatch"

PATCH_SIZE = 50

REFLECTANCE_RANGE = (0.0, 1.0)

LINE_STEP = 256

SAMPLES_ACROSS = 5

MAP = """Group = Mapping
  ProjectionName = {name}
  CenterLatitude = {latitude}
  CenterLongitude = {longitude}
  LongitudeDomain = 360
End_Group
End
"""

# How a scan is named, for its mission phase, orbit, latitude and where it looked.
NAMING = Naming(
    re.compile(
        r"^(?P<scan>(?:[a-z]\d{2}|moi)_\d{6}_\d{4}_[a-z]{2}_\d{2}[ns]\d{3}[we])$"
    ),
    identity="{scan}",
)

# What the arrays of one scan hold, and which of them is stored for.
LAYOUT = Layout(
    instrument="CTX",
    dims=("line", "sample"),
    axes=(GROUND, GROUND),
    measurement="image",
)

CACHE = ProductCache(paths.CTX_ROOT, {None: (CUBE_SUFFIX, METADATA_SUFFIX)})
