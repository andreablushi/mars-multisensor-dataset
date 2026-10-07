"""Reading one CRISM observation off disk, cleaning it, and joining its detectors."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism import clean
from building.preprocessing.crism.correction import (
    bands_calibration,
    merge,
    photometric,
)
from building.preprocessing.crism.models.observation import (
    ACQUISITION_PLANES,
    CrismObservation,
)
from common.pds import images, labels

# What a wavelength file writes where the detector was never calibrated.
UNCALIBRATED = 65535.0

# What a label says about the calibration software, the same in every product.
GROUND_SOFTWARE = ("MRO:IKF_", "MRO:RSC_", "MRO:REFZ_", "MRO:FRAM_STAT_")


def cached_detectors(identifier: str) -> tuple[configs.Detector, ...]:
    """Read which detectors of one observation were downloaded whole.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        detectors: The detectors whose observation landed, the first of them the one
            whose geometry places the observation.

    Raises:
        FileNotFoundError: When neither detector landed whole.
    """
    found = tuple(
        name
        for name in configs.Detector
        if all(
            path.exists()
            for path in configs.CACHE.product_files(
                identifier, configs.Kind.OBSERVATION, detector=name
            ).values()
        )
    )
    if not found:
        raise FileNotFoundError(f"No detector of {identifier} is in the cache.")
    return found


def detector_rows(image: Path, label: dict[str, str]) -> np.ndarray:
    """Return the detector row every band of one image was read off.

    Args:
        image: The `.img` file, which holds the row table after its cube.
        label: Its parsed label, which says where that table starts.

    Returns:
        rows: One detector row per band, in stored band order.
    """
    record = int(re.search(r"(\d+)\s*\)", label["^ROWNUM_TABLE"]).group(1))
    size = int(label["RECORD_BYTES"].split()[0])
    rows = np.fromfile(image, ">u2", int(label["BANDS"]), offset=(record - 1) * size)
    # Only the low nine bits number the row.
    return rows & 0x1FF


def read_transmission(
    label: dict[str, str], rows: np.ndarray, table: np.ndarray
) -> np.ndarray:
    """Read the atmosphere's transmission over the rows one infrared scan was read off.

    Args:
        label: The scan's label, which picks the record.
        rows: The detector row of every band of the scan, in stored order.
        table: The scan's centre wavelength of every column and band, stored order.

    Returns:
        transmission: Per column and band in wavelength order, NaN where unknown.

    Raises:
        ValueError: When the record lacks a row or does not fit the scan.
    """
    name = configs.transmission_record(label).lower()
    record = configs.CACHE.files(configs.WAVELENGTH_DIR, name)[".img"]
    values, held = images.load_cube(record)
    measured = detector_rows(record, held)
    at = np.searchsorted(measured, rows).clip(max=measured.size - 1)
    if (measured[at] != rows).any():
        raise ValueError(f"{name} holds no transmission for some rows of this scan.")
    picked = np.where(values[:, :, at] >= UNCALIBRATED, np.nan, values[:, :, at])
    return bands_calibration.calibrated_cube(picked, table)[0][0]


def read_detectors(
    identifier: str, found: tuple[configs.Detector, ...]
) -> tuple[
    dict[configs.Detector, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    list[dict[str, str]],
]:
    """Read every image one observation was downloaded as, keyed by detector.

    Args:
        identifier: The observation, its files already in the download cache.
        found: The detectors that landed whole.

    Returns:
        detectors: Each detector's cube, wavelength table and the atmosphere's
            transmission, None off the infrared, bands ascending.
        held: Each detector's observation label, in the order found.

    Raises:
        FileNotFoundError: When a wavelength or transmission file is missing.
        ValueError: When the band order, wavelength or transmission file cannot be read.
    """
    detectors = {}
    held = []
    for name in found:
        image = configs.CACHE.product_files(
            identifier, configs.Kind.OBSERVATION, detector=name
        )[".img"]
        cube, label = images.load_cube(image)
        held.append(label)
        # The wavelength file this half was calibrated against, and no other.
        wavelength = Path(label[configs.WAVELENGTH_KEY]).stem.lower()
        record = configs.CACHE.files(configs.WAVELENGTH_DIR, wavelength)[".img"]
        written = images.load_cube(record)[0][0]
        table = np.where(written >= UNCALIBRATED, np.nan, written.astype("f8"))
        transmission = None
        if name == configs.Detector.INFRARED:
            rows = detector_rows(image, label)
            transmission = read_transmission(label, rows, table)
        # Order the bands by wavelength and mark what was never calibrated.
        detectors[name] = (
            *bands_calibration.calibrated_cube(cube, table),
            transmission,
        )
    return detectors, held


def read_observation(identifier: str) -> CrismObservation:
    """Read one observation, clean it, and join the detectors it has onto the grid.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        observation: The observation on the survey's shared grid.

    Raises:
        FileNotFoundError: When any file the observation needs is missing.
        ValueError: When a window keeps no band of a cube.
    """
    found = cached_detectors(identifier)
    detectors, held = read_detectors(identifier, found)
    cleaned = clean.clean_detectors(detectors)
    geometry, geometry_label = images.load_cube(
        configs.CACHE.product_files(
            identifier, configs.Kind.GEOMETRY, detector=found[0]
        )[".img"]
    )
    label = {
        key: value
        for key, value in labels.merge(*held, geometry_label).items()
        if not key.startswith(GROUND_SOFTWARE)
    }
    observation = merge.merge_detectors(cleaned, geometry, label)
    valid = photometric.photometric_valid(
        observation.cube,
        observation.geometry[:, :, ACQUISITION_PLANES["incidence_deg"]],
        observation.valid,
        observation.measured_bands,
    )
    return replace(observation, valid=valid)
