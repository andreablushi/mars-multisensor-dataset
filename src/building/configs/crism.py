"""What a CRISM survey observation is published as, and where it lands."""

from __future__ import annotations

import re
from enum import StrEnum

from building import paths
from building.common.layout import Axis, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache


class Detector(StrEnum):
    """The two detectors of one scan, in the order a lone half places itself."""

    INFRARED = "l"
    VISIBLE = "s"


class Kind(StrEnum):
    """The two products one detector of a scan is published as."""

    OBSERVATION = "observation"
    GEOMETRY = "geometry"


# fmt: off
DETECTOR_BANDS_NM = {
    Detector.INFRARED: (
        1023.588, 1049.797, 1082.565, 1154.685, 1213.722, 1253.094, 1259.658,
        1266.221, 1279.350, 1331.876, 1371.284, 1377.853, 1384.423, 1390.993,
        1397.563, 1404.134, 1410.704, 1417.276, 1423.847, 1430.419, 1436.991,
        1443.564, 1450.137, 1456.710, 1463.284, 1469.857, 1476.432, 1483.006,
        1489.581, 1496.156, 1502.732, 1509.308, 1561.926, 1627.730, 1660.644,
        1693.566, 1713.323, 1733.084, 1752.847, 1779.203, 1812.155, 1838.522,
        1878.084, 1911.061, 1917.657, 1924.254, 1930.851, 1937.448, 1944.046,
        1950.644, 1957.242, 1963.841, 1970.440, 1977.039, 1983.639, 1990.239,
        2003.440, 2010.041, 2023.244, 2043.051, 2069.465, 2076.069, 2082.674,
        2089.279, 2095.884, 2102.490, 2109.096, 2115.702, 2122.309, 2128.916,
        2135.523, 2142.131, 2148.739, 2155.347, 2161.956, 2168.565, 2175.174,
        2181.783, 2188.393, 2195.004, 2201.614, 2208.225, 2214.836, 2221.448,
        2228.060, 2234.672, 2241.285, 2247.898, 2254.511, 2261.125, 2267.739,
        2274.353, 2280.967, 2287.582, 2294.197, 2300.813, 2307.429, 2314.045,
        2320.662, 2327.279, 2333.896, 2340.513, 2347.131, 2353.750, 2360.368,
        2366.987, 2373.606, 2380.226, 2386.846, 2393.466, 2400.086, 2406.707,
        2413.328, 2419.950, 2426.572, 2433.194, 2446.439, 2459.686, 2466.310,
        2472.934, 2479.559, 2486.183, 2492.809, 2499.434, 2506.031, 2512.629,
        2519.227, 2525.825, 2532.423, 2539.022, 2545.622, 2552.221, 2558.821,
        2585.225, 2605.032, 2624.842, 2631.447, 2644.656,
    ),
    Detector.VISIBLE: (
        402.231, 408.715, 415.200, 421.684, 428.170, 434.656, 441.142,
        447.629, 454.116, 460.604, 467.092, 473.581, 480.070, 486.559,
        493.049, 499.540, 506.031, 512.523, 519.014, 525.507, 532.000,
        538.493, 544.987, 551.482, 557.976, 564.472, 570.968, 577.464,
        583.961, 590.458, 596.955, 603.454, 609.952, 616.451, 622.951,
        629.451, 635.951, 642.452, 648.954, 655.456, 661.958, 668.461,
        674.965, 681.468, 687.973, 694.478, 700.983, 707.489, 713.995,
        720.502, 727.009, 733.516, 740.025, 746.533, 753.042, 759.552,
        766.062, 772.572, 779.083, 785.595, 792.107, 798.619, 805.132,
        811.645, 818.159, 824.673, 831.188, 837.703, 844.219, 850.735,
        857.252, 863.769, 870.287, 876.805, 883.323, 889.842, 896.362,
        902.882, 909.402, 915.923, 922.445, 928.966, 935.489, 942.012,
        948.535, 955.059, 961.583, 968.108, 974.633, 981.159, 987.685,
        994.211, 1000.738, 1007.266, 1013.794, 1020.323, 1026.852, 1033.381,
        1039.911, 1046.441, 1052.972,
    ),
}
# fmt: on

# The one band axis every observation is laid out on, both detectors in order.
BANDS_NM = tuple(sorted(band for grid in DETECTOR_BANDS_NM.values() for band in grid))

# Where each detector's bands sit along that axis.
DETECTOR_SLOTS = {
    detector: tuple(BANDS_NM.index(band) for band in bands)
    for detector, bands in DETECTOR_BANDS_NM.items()
}

# The nm window each detector is trusted over, outside which the reading is noise.
DETECTOR_WINDOWS_NM = {
    Detector.INFRARED: (1020.0, 2650.0),
    Detector.VISIBLE: (400.0, 1060.0),
}

# Where the atmosphere absorbs, in nm. Only the 2.0 um CO2 band is worth dropping.
ATMOSPHERIC_BANDS_NM = {Detector.INFRARED: ((1940.0, 2090.0),), Detector.VISIBLE: ()}

# How far above its column's mean a band reads as a spike, set per detector.
STRIPE_SIGMA = {Detector.INFRARED: 5.0, Detector.VISIBLE: 3.0}

# How ODE spells one detector; radiance and reflectance are the one observation
NAMING = Naming(
    re.compile(
        r"^(?P<stem>\w+)_(?:if|ra)(?P<code>\d+)(?P<detector>[ls]?)_(?P<level>trr\d+)$"
    ),
    identity="{stem}_if{code}_{level}",
    marks=("detector",),
    template="{stem}_{marker}{code}{detector}_{level}",
    fields={
        Kind.OBSERVATION: {"marker": "if"},
        Kind.GEOMETRY: {"marker": "de", "level": "ddr1"},
    },
)

# What the arrays of one observation hold, and which of them is stored for.
LAYOUT = Layout(
    instrument="CRISM",
    dims=("line", "sample", "band"),
    axes=(Axis.GROUND, Axis.GROUND, Axis.WAVELENGTH),
    measurement="cube",
    beside={
        "measured_bands": ("band",),
        "incidence_deg": ("line", "sample"),
        "emission_deg": ("line", "sample"),
        "phase_deg": ("line", "sample"),
        "local_solar_time_h": ("line", "sample"),
    },
    stored="f2",
    band_centres_nm=BANDS_NM,
)

# Where each product is kept, the geometry in a subdirectory beside its own scan.
CACHE = ProductCache(
    paths.CRISM_ROOT, NAMING, {None: (".lbl", ".img")}, {Kind.GEOMETRY: "ddr"}
)

# What a label calls the wavelength file it was calibrated against.
WAVELENGTH_KEY = "MRO:WAVELENGTH_FILE_NAME"

# The directory every wavelength file is kept in, shared by every observation.
WAVELENGTH_DIR = "cdr"
