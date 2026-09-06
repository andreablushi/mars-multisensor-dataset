"""What a SHARAD radargram is published as, and where it lands."""

from __future__ import annotations

import re

from building.common.layout import DELAY, GROUND, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache
from utils.disk import paths

# The two products one track is published as.
OBSERVATION = "observation"
GEOMETRY = "geometry"
KINDS = (OBSERVATION, GEOMETRY)

# How ODE spells one product of a track, which writes its kind after the track
# the observation on its own ends at.
NAMING = Naming(
    re.compile(r"^(?P<track>s_\d+)(?:_(?P<marker>rgram|geom))?$"),
    identity="{track}",
    marks=("marker",),
    template="{track}_{marker}",
    fields={OBSERVATION: {"marker": "rgram"}, GEOMETRY: {"marker": "geom"}},
)

# What the arrays of one track hold, and which of them is stored for. A sounder
# walks a line rather than sweeping ground, so only one axis is placed.
LAYOUT = Layout(
    instrument="SHARAD",
    dims=("delay", "trace"),
    axes=(DELAY, GROUND),
    measurement="power",
    beside={"traces": ("trace",)},
)

# Where each product of an observation is kept. The geometry is a table rather
# than an image, and sits in a subdirectory of its own.
CACHE = ProductCache(
    paths.SHARAD_ROOT,
    {OBSERVATION: (".lbl", ".img"), GEOMETRY: (".lbl", ".tab")},
    {GEOMETRY: "geom"},
)
