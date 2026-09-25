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


def product_files(
    identifier: str, detector: configs.Detector, kind: configs.Kind
) -> dict[str, Path]:
    """Return where each half of one detector's product of an observation belongs.

    Args:
        identifier: The observation the product is a part of.
        detector: Which detector's half of it.
        kind: Which product of that half, the observation or the geometry.

    Returns:
        files: The path for each suffix it is published as, keyed by suffix.
    """
    return configs.CACHE.files(
        identifier, configs.NAMING.product(identifier, kind, detector=detector), kind
    )


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
            for path in product_files(identifier, name, kind).values()
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
) -> dict[configs.Detector, tuple[np.ndarray, np.ndarray]]:
    """Read every image one observation was downloaded as, keyed by detector.

    Args:
        identifier: The observation, its files already in the download cache.
        found: The detectors that landed whole.

    Returns:
        detectors: Each detector's cube and wavelengths, bands ascending.

    Raises:
        FileNotFoundError: When a wavelength file is missing.
        ValueError: When the band order or wavelength file cannot be read.
    """
    detectors = {}
    for name in found:
        cube, label = images.load_cube(
            product_files(identifier, name, configs.Kind.OBSERVATION)[".img"]
        )
        # The wavelength file this half was calibrated against, and no other.
        wavelength = Path(label[configs.WAVELENGTH_KEY]).stem.lower()
        record = configs.CACHE.files(configs.WAVELENGTH_DIR, wavelength)[".img"]
        # Order the bands by wavelength and mark what was never calibrated.
        detectors[name] = bands_calibration.calibrate(cube, read_wavelengths(record))
    return detectors


def read_label(identifier: str, found: tuple[configs.Detector, ...]) -> dict[str, str]:
    """Read what every product one observation is published as says about it.

    Args:
        identifier: The observation, its files already in the download cache.
        found: The detectors that landed whole, the first of them placing it.

    Returns:
        label: Their labels merged, without the calibration software's tuning.

    Raises:
        FileNotFoundError: When a label is missing.
    """
    held = [
        labels.load(product_files(identifier, name, configs.Kind.OBSERVATION)[".lbl"])
        for name in found
    ]
    held.append(
        labels.load(product_files(identifier, found[0], configs.Kind.GEOMETRY)[".lbl"])
    )
    return {
        key: value
        for key, value in labels.merge(*held).items()
        if not key.startswith(GROUND_SOFTWARE)
    }


def read_geometry(identifier: str, placing: configs.Detector) -> np.ndarray:
    """Read the backplanes that place every pixel of one observation.

    Args:
        identifier: The observation, its files already in the download cache.
        placing: The detector whose geometry places the observation.

    Returns:
        backplanes: That detector's backplanes, lines by samples by 14.

    Raises:
        FileNotFoundError: When the geometry or its label is missing.
        KeyError: When the label names a sample type this cannot read.
    """
    return images.load_cube(
        product_files(identifier, placing, configs.Kind.GEOMETRY)[".img"]
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
    return merge.merge_detectors(
        identifier,
        clean.clean_detectors(identifier, read_detectors(identifier, found)),
        read_geometry(identifier, found[0]),
        read_label(identifier, found),
    )
