"""What a SHARAD radargram is published as, and where it lands."""

from __future__ import annotations

import re

from building import paths
from building.common.layout import DELAY, GROUND, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache

# The three products one track is published as.
OBSERVATION = "observation"
GEOMETRY = "geometry"
CLUTTER = "clutter"
KINDS = (OBSERVATION, GEOMETRY, CLUTTER)

# How ODE spells one product of a track, its kind written after the track itself.
NAMING = Naming(
    re.compile(r"^(?P<track>s_\d+)(?:_(?P<marker>rgram|geom|sim))?$"),
    identity="{track}",
    marks=("marker",),
    template="{track}_{marker}",
    fields={
        OBSERVATION: {"marker": "rgram"},
        GEOMETRY: {"marker": "geom"},
        CLUTTER: {"marker": "sim"},
    },
)

# What one track's arrays hold. A sounder walks a line, so only one axis is placed.
LAYOUT = Layout(
    instrument="SHARAD",
    dims=("delay", "trace"),
    axes=(DELAY, GROUND),
    measurement="power",
    beside={
        "traces": ("trace",),
        "clutter": ("delay", "trace"),
        "incidence_deg": ("trace",),
        "spacecraft_altitude_km": ("trace",),
    },
)

DELAY_INTERVAL_S = 0.0375e-6

AREOID_ROW = 1800

DELAY_ROWS = 3600

# Where each product is kept. The geometry and the clutter each in a subdirectory.
CACHE = ProductCache(
    paths.SHARAD_ROOT,
    {OBSERVATION: (".lbl", ".img"), GEOMETRY: (".lbl", ".tab"), CLUTTER: (".img",)},
    {GEOMETRY: "geom", CLUTTER: "sim"},
)

CLUTTER_ARRAY = 2

CLUTTER_TYPE = "<f4"
