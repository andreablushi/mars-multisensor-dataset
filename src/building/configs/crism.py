"""What a CRISM multispectral survey observation is published as, and where it lands."""

from __future__ import annotations

import re

from building.common.layout import GROUND, WAVELENGTH, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache
from utils.disk import paths

# The two detectors of one scan, infrared and visible.
DETECTORS = ("l", "s")

# The two products one detector of a scan is published as.
OBSERVATION = "observation"
GEOMETRY = "geometry"
KINDS = (OBSERVATION, GEOMETRY)

# How ODE spells one detector of an observation, the detector and the kind
# written where the observation's own id carries neither.
NAMING = Naming(
    re.compile(r"^(?P<stem>\w+)_if(?P<code>\d+)(?P<detector>[ls]?)_(?P<level>trr\d+)$"),
    identity="{stem}_if{code}_{level}",
    marks=("detector",),
    template="{stem}_{marker}{code}{detector}_{level}",
    fields={
        OBSERVATION: {"marker": "if"},
        GEOMETRY: {"marker": "de", "level": "ddr1"},
    },
)

# What a label calls the wavelength file it was calibrated against.
WAVELENGTH_KEY = "MRO:WAVELENGTH_FILE_NAME"

# Where each product of an observation is kept, and what it is called there.
# The geometry sits in a subdirectory of its own, beside the scan it belongs to.
CACHE = ProductCache(paths.CRISM_ROOT, {None: (".lbl", ".img")}, {GEOMETRY: "ddr"})

# What the arrays of one observation hold, and which of them is stored for.
LAYOUT = Layout(
    instrument="CRISM",
    dims=("line", "sample", "band"),
    axes=(
        GROUND,
        GROUND,
        WAVELENGTH,
    ),
    measurement="cube",
    beside={"wavelengths": ("sample", "band"), "columns": ("sample",)},
)

# The directory every wavelength file is kept in, shared by every observation.
WAVELENGTH_DIR = "cdr"
