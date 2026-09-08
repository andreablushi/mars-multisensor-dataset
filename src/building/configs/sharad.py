"""What a SHARAD radargram is published as, and where it lands."""

from __future__ import annotations

import re

from building import paths
from building.common.layout import ELEVATION, GROUND, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache

# The two products one track is published as.
OBSERVATION = "observation"
GEOMETRY = "geometry"
KINDS = (OBSERVATION, GEOMETRY)

# How ODE spells one product of a track, its kind written after the track itself.
NAMING = Naming(
    re.compile(r"^(?P<track>s_\d+)(?:_(?P<marker>rgram|geom))?$"),
    identity="{track}",
    marks=("marker",),
    template="{track}_{marker}",
    fields={OBSERVATION: {"marker": "rgram"}, GEOMETRY: {"marker": "geom"}},
)

# What one track's arrays hold. A sounder walks a line, so only one axis is placed.
LAYOUT = Layout(
    instrument="SHARAD",
    dims=("delay", "trace"),
    axes=(ELEVATION, GROUND),
    measurement="power",
    beside={"traces": ("trace",), "elevation": ("delay",)},
)

# Where each product is kept. The geometry is a table, in a subdirectory of its own.
CACHE = ProductCache(
    paths.SHARAD_ROOT,
    {OBSERVATION: (".lbl", ".img"), GEOMETRY: (".lbl", ".tab")},
    {GEOMETRY: "geom"},
)
