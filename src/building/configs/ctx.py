"""What a CTX scan is published as, and where it lands."""

from __future__ import annotations

import re

from building.common.layout import GROUND, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache
from utils.disk import paths

# The two products a scan comes as, the label first so a wrong projection costs less.
LABEL = "label"
IMAGE = "image"
KINDS = (LABEL, IMAGE)

# What each kind is suffixed with once it is on disk.
SUFFIXES = {LABEL: ".isis.hdr", IMAGE: ".tiff"}

# From this latitude up ASU writes a scan on a polar stereographic grid, and
# below it on a simple cylindrical one. Measured against ASU over the selection:
# every scan from 63 to 69 is cylindrical and every one from 70 up is polar.
POLAR_LATITUDE = 70

# Where a scan's own name carries the latitude it was taken at.
LATITUDE = re.compile(r"_(\d{2})[ns]\d{3}[we]$")


def polar(identifier: str) -> bool:
    """Say whether ASU writes one scan on a polar grid.

    Args:
        identifier: The scan, whose name carries the latitude it was taken at.

    Returns:
        True where ASU projects it stereographically, and False where it does
        not, which is also the answer for a name carrying no latitude.
    """
    found = LATITUDE.search(identifier.lower())
    return bool(found) and int(found[1]) >= POLAR_LATITUDE


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

# Where both are kept. ASU names both after the scan, so only the suffix differs.
CACHE = ProductCache(paths.CTX_ROOT, {None: tuple(SUFFIXES.values())})
