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
