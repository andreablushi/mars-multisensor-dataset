"""What a CRISM survey observation is published as, and where it lands."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from building import paths
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


# The nm window each detector is trusted over, outside which the reading is noise.
DETECTOR_WINDOWS_NM = {
    Detector.INFRARED: (1020.0, 2650.0),
    Detector.VISIBLE: (400.0, 1060.0),
}

# Where the atmosphere absorbs, in nm. Only the 2.0 um CO2 band is worth dropping.
CO2_WINDOW_NM = (1940.0, 2090.0)

CO2_BAND_NM = (2007.0, 1980.0)

NOISY_BANDS_NM = (648.954, 1052.972, 2631.447)

# What a wavelength file writes where the detector was never calibrated.
UNCALIBRATED = 65535.0

FILL = 0.0

CALIBRATION_ROOT = Path(__file__).parent / "calibration"

WAVELENGTH_FILES = {
    Detector.INFRARED: CALIBRATION_ROOT / "cdr490947778566_wa0300010l_3.img",
    Detector.VISIBLE: CALIBRATION_ROOT / "cdr450924300802_wa0300010s_2.img",
}

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

# What ODE publishes CRISM under.
ODE = {"ihid": "MRO", "iid": "CRISM"}

# The ODE product types an observation and its geometry are published under.
PRODUCT_TYPES = {Kind.OBSERVATION: "TRDR", Kind.GEOMETRY: "DDR"}

# Where each product is kept, the geometry in a subdirectory beside its own scan.
CACHE = ProductCache(
    paths.CRISM_ROOT, NAMING, {None: (".lbl", ".img")}, {Kind.GEOMETRY: "ddr"}
)
