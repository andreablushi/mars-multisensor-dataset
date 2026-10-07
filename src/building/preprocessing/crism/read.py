"""Reading one CRISM observation off disk, cleaning it, and joining its detectors."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism import clean, merge
from building.preprocessing.crism.models.observation import (
    ACQUISITION_PLANES,
    CrismObservation,
)
from common.pds import images, labels


def calibrated_cube(
    cube: np.ndarray, table: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Order one cube by wavelength and fill what was never calibrated.

    Args:
        cube: The values as lines by samples by bands, in stored band order.
        table: The centre wavelength of every column and band, NaN if uncalibrated.

    Returns:
        cube: The cube with bands ascending, uncalibrated columns and bands NaN.
        table: The centre wavelength of every column and band, in that same order.

    Raises:
        ValueError: When the table does not fit the cube or has no calibrated band.
    """
    if cube.shape[1:] != table.shape:
        raise ValueError(
            f"A cube of {cube.shape[1]} columns by {cube.shape[2]} bands cannot "
            f"be read with a table of {table.shape[0]} by {table.shape[1]}."
        )
    # What every band of this detector is centred on, averaged over its columns.
    centres = configs.band_centres(table)
    # Read the direction off the file instead of assuming one.
    named = np.flatnonzero(~np.isnan(centres))
    if not named.size:
        raise ValueError("No band of this cube was ever calibrated.")
    if centres[named[0]] > centres[named[-1]]:
        cube, table = cube[:, :, ::-1], table[:, ::-1]

    # A writable copy in the new order, since the reversal above is a view.
    ordered = np.array(cube, dtype="f4")
    # Say what was never calibrated with NaN, leaving the shape alone.
    blank = np.isnan(table)
    ordered[:, blank.all(axis=1), :] = np.nan
    ordered[:, :, blank.all(axis=0)] = np.nan
    return ordered, table


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


def read_atmosphere_transmission(
    label: dict[str, str], rows: np.ndarray, table: np.ndarray
) -> np.ndarray:
    """Read the atmosphere's transmission used by the volcano scan.

    Args:
        label: The scan's label, which picks the record.
        rows: The detector row of every band of the scan, in stored order.
        table: The scan's centre wavelength of every column and band, stored order.

    Returns:
        transmission: Per column and band in wavelength order, NaN where unknown.
    """
    name = configs.transmission_record(label).lower()
    record = configs.CACHE.files(configs.WAVELENGTH_DIR, name)[".img"]
    values, held = images.load_cube(record)
    band_of_row = {row: band for band, row in enumerate(detector_rows(record, held))}
    at = [band_of_row[row] for row in rows]
    picked = np.where(
        values[:, :, at] >= configs.UNCALIBRATED, np.nan, values[:, :, at]
    )
    return calibrated_cube(picked, table)[0][0]


def read_detectors(
    identifier: str, found: list[configs.Detector]
) -> tuple[
    dict[configs.Detector, tuple[np.ndarray, np.ndarray]],
    np.ndarray | None,
    list[dict[str, str]],
]:
    """Read every image one observation was downloaded as, keyed by detector.

    Args:
        identifier: The observation, its files already in the download cache.
        found: The detectors that landed, the first of them the one whose geometry
            places the observation.

    Returns:
        detectors: Each detector's cube and wavelength table, bands ascending.
        transmission: The atmosphere's transmission over the infrared scan, or None
            when the observation has no infrared half.
        held: Each detector's observation label, in the order found.

    Raises:
        FileNotFoundError: When a wavelength or transmission file is missing.
        ValueError: When the band order or wavelength file cannot be read.
    """
    detectors = {}
    transmission = None
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
        table = configs.wavelength_table(record)
        if name == configs.Detector.INFRARED:
            rows = detector_rows(image, label)
            transmission = read_atmosphere_transmission(label, rows, table)
        # Order the bands by wavelength and mark what was never calibrated.
        detectors[name] = calibrated_cube(cube, table)
    return detectors, transmission, held


def read_observation(identifier: str) -> CrismObservation | None:
    """Read one observation, clean it, and join the detectors it has onto the grid.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        observation: The observation on the survey's shared grid, or None where no
            detector measured a pixel.

    Raises:
        FileNotFoundError: When any file the observation needs is missing.
        ValueError: When a window keeps no band of a cube.
    """
    found = [
        name
        for name in configs.Detector
        if configs.CACHE.product_files(
            identifier, configs.Kind.OBSERVATION, detector=name
        )[".img"].exists()
    ]
    if not found:
        raise FileNotFoundError(f"No detector of {identifier} is in the cache.")
    detectors, transmission, held = read_detectors(identifier, found)
    cleaned = clean.clean_detectors(detectors, transmission)
    if dropped := [name for name in found if name not in cleaned]:
        missed = ", ".join(dropped)
        print(f"note {identifier} [CRISM]: no pixel measured on {missed}", flush=True)
    if not cleaned:
        return None
    geometry, geometry_label = images.load_cube(
        configs.CACHE.product_files(
            identifier, configs.Kind.GEOMETRY, detector=found[0]
        )[".img"]
    )
    label = labels.merge(*held, geometry_label)
    observation = merge.merge_detectors(cleaned, geometry, label)
    valid = clean.photometric_valid(
        observation.cube,
        observation.geometry[:, :, ACQUISITION_PLANES["incidence_deg"]],
        observation.valid,
        observation.measured_bands,
    )
    return replace(observation, valid=valid)
