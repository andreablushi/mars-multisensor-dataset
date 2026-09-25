"""What a SHARAD radargram is published as, and where it lands."""

from __future__ import annotations

import re
from enum import StrEnum

from building import paths
from building.common.layout import Axis, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache


class Kind(StrEnum):
    """The three products one track is published as."""

    OBSERVATION = "observation"
    GEOMETRY = "geometry"
    CLUTTER = "clutter"


# How ODE spells one product of a track, its kind written after the track itself.
NAMING = Naming(
    re.compile(r"^(?P<track>s_\d+)(?:_(?P<marker>rgram|geom|sim))?$"),
    identity="{track}",
    marks=("marker",),
    template="{track}_{marker}",
    fields={
        Kind.OBSERVATION: {"marker": "rgram"},
        Kind.GEOMETRY: {"marker": "geom"},
        Kind.CLUTTER: {"marker": "sim"},
    },
)

# What one track's arrays hold. A sounder walks a line, so only one axis is placed.
LAYOUT = Layout(
    instrument="SHARAD",
    dims=("delay", "trace"),
    axes=(Axis.DELAY, Axis.GROUND),
    measurement="power",
    beside={
        "traces": ("trace",),
        "clutter": ("delay", "trace"),
        "incidence_deg": ("trace",),
        "spacecraft_altitude_km": ("trace",),
    },
)

# Where each product is kept. The geometry and the clutter each in a subdirectory.
CACHE = ProductCache(
    paths.SHARAD_ROOT,
    {
        Kind.OBSERVATION: (".lbl", ".img"),
        Kind.GEOMETRY: (".lbl", ".tab"),
        Kind.CLUTTER: (".img",),
    },
    {Kind.GEOMETRY: "geom", Kind.CLUTTER: "sim"},
)

DELAY_INTERVAL_S = 0.0375e-6

AREOID_ROW = 1800

DELAY_ROWS = 3600

CLUTTER_ARRAY = 2

CLUTTER_TYPE = "<f4"
