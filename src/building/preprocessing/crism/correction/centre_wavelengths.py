"""Aligning each scan to the fixed wavelength calibration and its joining grid."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from building.common.layout import Axis, Layout
from building.configs import crism as configs
from common.pds import images, labels


def detector_rows(image: Path, label: dict[str, str]) -> np.ndarray:
    """Return the detector row every band of one image was read off.

    Args:
        image: The `.img` file, which holds the row table right after its cube.
        label: Its parsed label, which sizes that cube.

    Returns:
        rows: One detector row per band, in stored band order.
    """
    lines, samples, bands, _, dtype = labels.image_layout(label)
    cube_bytes = lines * samples * bands * np.dtype(dtype).itemsize
    return np.fromfile(image, ">u2", bands, offset=cube_bytes)


def read_calibration(record: Path, rows: np.ndarray | None = None) -> np.ndarray:
    """Return a calibration file's values per column at the given detector rows.

    Args:
        record: The calibration file's `.img`, its `.lbl` beside it.
        rows: The detector row of every band wanted, or None for every band it holds.

    Returns:
        values: Per column and band in the order of the rows, NaN if uncalibrated.
    """
    values, label = images.load_cube(record)
    stored = detector_rows(record, label)
    band_of_row = {row: band for band, row in enumerate(stored)}
    picked = values[0][
        :, [band_of_row[row] for row in (stored if rows is None else rows)]
    ]
    return np.where(picked >= configs.UNCALIBRATED, np.nan, picked)


def band_centres(table: np.ndarray) -> np.ndarray:
    """Return the centre wavelength of every band, averaged over its columns.

    Args:
        table: The centre wavelength of every column and band.

    Returns:
        centres: One centre per band averaged over its columns, NaN where none.
    """
    with warnings.catch_warnings(action="ignore"):
        return np.nanmean(table, axis=0)


def window_bands(table: np.ndarray, detector: configs.Detector) -> np.ndarray:
    """Return the bands a detector is trusted over, ascending in wavelength.

    Args:
        table: The centre wavelength of every column and band.
        detector: Which detector, `l` for infrared or `s` for visible.

    Returns:
        bands: The index of every band inside the detector's window.
    """
    centres = band_centres(table)
    low, high = configs.DETECTOR_WINDOWS_NM[detector]
    inside = np.flatnonzero((centres >= low) & (centres <= high))
    return inside[np.argsort(centres[inside])]


def outside_co2(centres: np.ndarray) -> np.ndarray:
    """Return which bands lie outside the CO2 window.

    Args:
        centres: The centre wavelength of every band.

    Returns:
        outside: One flag per band, True where the atmosphere leaves the ground.
    """
    low, high = configs.CO2_WINDOW_NM
    return (centres < low) | (centres > high)


def survey_bands_nm(detector: configs.Detector) -> tuple[float, ...]:
    """Return the bands one detector is read onto, from the survey's wavelength file.

    Args:
        detector: Which detector, `l` for infrared or `s` for visible.

    Returns:
        bands: The centre of every survey band its window, the atmosphere and the
            noise leave, in nm, ascending.
    """
    table = read_calibration(configs.WAVELENGTH_FILES[detector]).astype("f8")
    centres = band_centres(table)[window_bands(table, detector)]
    kept = outside_co2(centres)
    kept &= ~np.isclose(centres[:, None], configs.NOISY_BANDS_NM, atol=1e-3).any(axis=1)
    return tuple(centres[kept].tolist())


DETECTOR_BANDS_NM = {
    detector: survey_bands_nm(detector) for detector in configs.Detector
}

# The one band axis every observation is laid out on, both detectors in order.
BANDS_NM = tuple(sorted(band for grid in DETECTOR_BANDS_NM.values() for band in grid))

# Where each detector's bands sit along that axis.
DETECTOR_SLOTS = {
    detector: [BANDS_NM.index(band) for band in bands]
    for detector, bands in DETECTOR_BANDS_NM.items()
}

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
