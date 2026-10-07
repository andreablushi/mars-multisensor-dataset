"""What a CRISM survey observation is published as, and where it lands."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

import numpy as np

from building import paths
from building.common.layout import Axis, Layout
from building.common.naming import Naming
from building.common.product_cache import ProductCache
from common.paths import CONFIGS_ROOT
from common.pds import images


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
ATMOSPHERIC_BANDS_NM = {Detector.INFRARED: ((1940.0, 2090.0),), Detector.VISIBLE: ()}

NOISY_BANDS_NM = (648.954, 1052.972, 2631.447)

# What a wavelength file writes where the detector was never calibrated.
UNCALIBRATED = 65535.0

SURVEY_WAVELENGTHS = {
    Detector.INFRARED: CONFIGS_ROOT / "crism" / "cdr490947778566_wa0300010l_3.img",
    Detector.VISIBLE: CONFIGS_ROOT / "crism" / "cdr450924300802_wa0300010s_2.img",
}


def band_centres(table: np.ndarray) -> np.ndarray:
    """Return the centre wavelength of every band, averaged over its columns.

    Args:
        table: The centre wavelength of every column and band.

    Returns:
        centres: One centre per band averaged over its columns, NaN where none.
    """
    # Bands the detector was calibrated for in at least one column.
    named = ~np.isnan(table).all(axis=0)
    # Averaging only those avoids taking the mean of an empty slice.
    out = np.full(table.shape[1], np.nan)
    out[named] = np.nanmean(table[:, named], axis=0)
    return out


def wavelength_table(record: Path) -> np.ndarray:
    """Return the centre wavelength of every column and band one wavelength file holds.

    Args:
        record: The wavelength file's `.img`, its `.lbl` beside it.

    Returns:
        table: The centre wavelength in nm per column and band, NaN if uncalibrated.
    """
    written = images.load_cube(record)[0][0]
    return np.where(written >= UNCALIBRATED, np.nan, written.astype("f8"))


def survey_bands_nm(detector: Detector) -> tuple[float, ...]:
    """Return the bands one detector is read onto, from the survey's wavelength file.

    Args:
        detector: Which detector, `l` for infrared or `s` for visible.

    Returns:
        bands: The centre of every survey band its window, the atmosphere and the
            noise leave, in nm, ascending.
    """
    centres = band_centres(wavelength_table(SURVEY_WAVELENGTHS[detector]))
    low, high = DETECTOR_WINDOWS_NM[detector]
    kept = (centres >= low) & (centres <= high)
    for start, stop in ATMOSPHERIC_BANDS_NM[detector]:
        kept &= (centres < start) | (centres > stop)
    kept &= ~np.isclose(centres[:, None], NOISY_BANDS_NM, atol=1e-3).any(axis=1)
    return tuple(sorted(centres[kept].tolist()))


DETECTOR_BANDS_NM = {detector: survey_bands_nm(detector) for detector in Detector}

# The one band axis every observation is laid out on, both detectors in order.
BANDS_NM = tuple(sorted(band for grid in DETECTOR_BANDS_NM.values() for band in grid))

# Where each detector's bands sit along that axis.
DETECTOR_SLOTS = {
    detector: tuple(BANDS_NM.index(band) for band in bands)
    for detector, bands in DETECTOR_BANDS_NM.items()
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

# Atmospheric transmission: every 10-column AT CDR APL made over Olympus Mons, by period
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
        record: The record's product id, the last for a scan after every period.

    Raises:
        ValueError: When the scan started before every period.
    """
    clock = float(label["SPACECRAFT_CLOCK_START_COUNT"].split("/")[1])
    started = [one for one in TRANSMISSION_RECORDS if int(one[5:15]) <= clock]
    if not started:
        raise ValueError(f"No transmission record covers a scan started at {clock}.")
    return started[-1]
