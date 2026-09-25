"""Reading one CRISM observation off disk, cleaning it, and joining its detectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism import clean
from building.preprocessing.crism.correction import bands_calibration, merge
from building.preprocessing.crism.models.observation import CrismObservation
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
        detectors: The detectors whose observation and geometry both landed, the
            first of them the one that places the observation.

    Raises:
        FileNotFoundError: When neither detector landed whole.
    """
    found = tuple(
        name
        for name in configs.Detector
        if all(
            path.exists()
            for kind in configs.Kind
            for path in configs.CACHE.product_files(
                identifier, kind, detector=name
            ).values()
        )
    )
    if not found:
        raise FileNotFoundError(f"No detector of {identifier} is in the cache.")
    return found


def read_wavelengths(record: Path) -> np.ndarray:
    """Read the centre wavelength of every column and band one record holds.

    Args:
        record: The `.img` of a wavelength file, which holds a single line.

    Returns:
        wavelengths: The centre wavelength in nm as columns by bands.
    """
    written = images.load_cube(record)[0][0]
    return np.where(written >= UNCALIBRATED, np.nan, written.astype("f8"))


def read_detectors(
    identifier: str, found: tuple[configs.Detector, ...]
) -> tuple[dict[configs.Detector, tuple[np.ndarray, np.ndarray]], list[dict[str, str]]]:
    """Read every image one observation was downloaded as, keyed by detector.

    Args:
        identifier: The observation, its files already in the download cache.
        found: The detectors that landed whole.

    Returns:
        detectors: Each detector's cube and wavelengths, bands ascending.
        held: Each detector's observation label, in the order found.

    Raises:
        FileNotFoundError: When a wavelength file is missing.
        ValueError: When the band order or wavelength file cannot be read.
    """
    detectors = {}
    held = []
    for name in found:
        cube, label = images.load_cube(
            configs.CACHE.product_files(
                identifier, configs.Kind.OBSERVATION, detector=name
            )[".img"]
        )
        held.append(label)
        # The wavelength file this half was calibrated against, and no other.
        wavelength = Path(label[configs.WAVELENGTH_KEY]).stem.lower()
        record = configs.CACHE.files(configs.WAVELENGTH_DIR, wavelength)[".img"]
        # Order the bands by wavelength and mark what was never calibrated.
        detectors[name] = bands_calibration.calibrated_cube(
            cube, read_wavelengths(record)
        )
    return detectors, held


def read_label(
    identifier: str, placing: configs.Detector, held: list[dict[str, str]]
) -> dict[str, str]:
    """Read what every product one observation is published as says about it.

    Args:
        identifier: The observation, its files already in the download cache.
        placing: The detector that places it, whose geometry label is read.
        held: Each detector's observation label, the placing one first.

    Returns:
        label: Their labels merged, without the calibration software's tuning.

    Raises:
        FileNotFoundError: When the geometry label is missing.
    """
    geometry = labels.load(
        configs.CACHE.product_files(
            identifier, configs.Kind.GEOMETRY, detector=placing
        )[".lbl"]
    )
    return {
        key: value
        for key, value in labels.merge(*held, geometry).items()
        if not key.startswith(GROUND_SOFTWARE)
    }


def read_geometry(identifier: str, placing: configs.Detector) -> np.ndarray:
    """Read the backplanes that place every pixel of one observation."""
    return images.load_cube(
        configs.CACHE.product_files(
            identifier, configs.Kind.GEOMETRY, detector=placing
        )[".img"]
    )[0]


def read_observation(identifier: str) -> CrismObservation:
    """Read one observation, clean it, and join the detectors it has onto the grid.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        observation: The observation on the survey's shared grid.

    Raises:
        FileNotFoundError: When any file the observation needs is missing.
        ValueError: When a window keeps no band of a cube, or no detector measured.
    """
    found = cached_detectors(identifier)
    detectors, held = read_detectors(identifier, found)
    return merge.merge_detectors(
        identifier,
        clean.clean_detectors(identifier, detectors),
        read_geometry(identifier, found[0]),
        read_label(identifier, found[0], held),
    )
