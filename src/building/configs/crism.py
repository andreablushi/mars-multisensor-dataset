"""What a CRISM multispectral survey observation is published as, and where it lands."""

from __future__ import annotations

import re
from itertools import chain

from building import paths
from building.common.layout import GROUND, WAVELENGTH, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache

# The two detectors of one scan, infrared and visible.
DETECTORS = ("l", "s")

# fmt: off
DETECTOR_BANDS_NM = {
    "l": (
        1023.588, 1049.797, 1082.565, 1154.685, 1213.722, 1253.094, 1259.658,
        1266.221, 1279.350, 1331.876, 1371.284, 1397.563, 1430.419, 1469.857,
        1502.732, 1509.308, 1561.926, 1627.730, 1660.644, 1693.566, 1752.847,
        1812.155, 1878.084, 1930.851, 1977.039, 1983.639, 2010.041, 2069.465,
        2122.309, 2142.131, 2168.565, 2208.225, 2234.672, 2254.511, 2294.197,
        2320.662, 2333.896, 2353.750, 2393.466, 2433.194, 2459.686, 2532.423,
        2605.032, 2631.447,
    ),
    "s": (
        408.715, 441.142, 532.000, 596.955, 648.954, 681.468, 707.489, 740.025,
        772.572, 798.619, 831.188, 857.252, 889.842, 922.445, 948.535, 981.159,
        1020.323, 1052.972,
    ),
}
# fmt: on

WAVELENGTHS_NM = tuple(sorted(chain.from_iterable(DETECTOR_BANDS_NM.values())))

BANDS = len(WAVELENGTHS_NM)

# The two products one detector of a scan is published as.
OBSERVATION = "observation"
GEOMETRY = "geometry"
KINDS = (OBSERVATION, GEOMETRY)

# How ODE spells one detector; radiance and reflectance are the one observation
NAMING = Naming(
    re.compile(
        r"^(?P<stem>\w+)_(?:if|ra)(?P<code>\d+)(?P<detector>[ls]?)_(?P<level>trr\d+)$"
    ),
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

# Where each product is kept, the geometry in a subdirectory beside its own scan.
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
)

# The directory every wavelength file is kept in, shared by every observation.
WAVELENGTH_DIR = "cdr"
