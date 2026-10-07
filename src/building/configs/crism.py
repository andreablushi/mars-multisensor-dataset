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
        1266.221, 1279.350, 1331.876, 1371.284, 1397.563, 1430.419, 1469.857,
        1502.732, 1509.308, 1561.926, 1627.730, 1660.644, 1693.566, 1752.847,
        1812.155, 1878.084, 1930.851, 2122.309, 2142.131, 2168.565, 2208.225,
        2234.672, 2254.511, 2294.197, 2320.662, 2333.896, 2353.750, 2393.466,
        2433.194, 2459.686, 2532.423, 2605.032,
    ),
    Detector.VISIBLE: (
        408.715, 441.142, 532.000, 596.955, 681.468, 707.489, 740.025, 772.572,
        798.619, 831.188, 857.252, 889.842, 922.445, 948.535, 981.159, 1020.323,
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

# What ODE publishes CRISM under.
ODE = {"ihid": "MRO", "iid": LAYOUT.instrument}

# The ODE product types an observation and its geometry are published under.
PRODUCT_TYPES = {Kind.OBSERVATION: "TRDR", Kind.GEOMETRY: "DDR"}

# The ODE product type a wavelength file is published under.
WAVELENGTH_PRODUCT_TYPE = "CDR"

# Where each product is kept, the geometry in a subdirectory beside its own scan.
CACHE = ProductCache(
    paths.CRISM_ROOT, NAMING, {None: (".lbl", ".img")}, {Kind.GEOMETRY: "ddr"}
)

# What a label calls the wavelength file it was calibrated against.
WAVELENGTH_KEY = "MRO:WAVELENGTH_FILE_NAME"

# The directory every wavelength file is kept in, shared by every observation.
WAVELENGTH_DIR = "cdr"

TRANSMISSION_RECORDS = (
    "CDR420843667218_AT0300000L_7",
    "CDR420845919018_AT0300000L_7",
    "CDR420853786818_AT0300000L_7",
    "CDR420858623419_AT0300000L_7",
    "CDR420862848019_AT0300000L_7",
    "CDR420865293319_AT0300000L_7",
    "CDR420869443219_AT0300000L_7",
    "CDR430873156619_AT0300000L_7",
    "CDR430876940219_AT0300000L_7",
    "CDR430880416919_AT0300000L_7",
    "CDR430886781719_AT0300000L_7",
    "CDR430887634919_AT0300000L_7",
    "CDR440891104419_AT0300000L_7",
    "CDR440895915820_AT0300000L_7",
    "CDR440903304820_AT0300000L_7",
    "CDR440914248820_AT0300000L_7",
    "CDR450920541621_AT0300000L_7",
    "CDR460929030422_AT0300000L_7",
)

CO2_BAND_NM = (2007.0, 1980.0)


def transmission_record(label: dict[str, str]) -> str:
    """Return the transmission record whose period holds one scan's start.

    Args:
        label: The scan's label, which says when it started.

    Returns:
        record: The record's product id, the first for a scan before any period.
    """
    clock = float(label["SPACECRAFT_CLOCK_START_COUNT"].split("/")[1])
    started = [one for one in TRANSMISSION_RECORDS if int(one[5:15]) <= clock]
    return started[-1] if started else TRANSMISSION_RECORDS[0]
