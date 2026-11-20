"""Reading one SHARAD track off disk, its geometry joined onto the radargram."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mars_multisensor_dataset.building.configs import sharad as configs
from mars_multisensor_dataset.building.models.settings import Preprocessing
from mars_multisensor_dataset.building.preprocessing.sharad.models.observation import (
    SharadObservation,
)
from mars_multisensor_dataset.building.preprocessing.sharad.normalize import (
    clutter_db,
    radargram_db,
)
from mars_multisensor_dataset.common.pds import images, labels, tables


def read_observation(
    identifier: str, preprocessing: Preprocessing
) -> SharadObservation:
    """Read one radargram and join it to the geometry it was measured at.

    Args:
        identifier: The observation, its files already in the download cache.
        preprocessing: The build's reader settings, of which SHARAD reads its gate.

    Returns:
        observation: The radargram and its clutter simulation in dB, with their
            geometry and the gate its crops' noise is measured with.

    Raises:
        FileNotFoundError: When any product or a label is missing.
        KeyError: When a label names a sample type this cannot read.
        ValueError: When the geometry is not one row per trace, or the clutter does
            not fit.
    """
    held = {
        kind: configs.CACHE.product_files(identifier, kind) for kind in configs.Kind
    }
    # The echoes themselves, then the places they were sounded at.
    power, sounding = images.load_cube(held[configs.Kind.OBSERVATION][".img"])
    power = power[:, :, 0]
    geometry, geometry_label = tables.load_table(held[configs.Kind.GEOMETRY][".tab"])
    if len(geometry) != power.shape[1]:
        raise ValueError(
            f"{identifier} places {len(geometry)} traces of {power.shape[1]}."
        )
    clutter = clutter_power(held[configs.Kind.CLUTTER][".img"], power.shape)
    return SharadObservation(
        labels.merge(sounding, geometry_label),
        radargram_db(power),
        clutter_db(clutter),
        geometry,
        preprocessing.SHARAD,
    )


def clutter_power(path: Path, shape: tuple[int, ...]) -> np.memmap:
    """Map one clutter simulation off disk, on its radargram's grid.

    Args:
        path: The simulation's image file.
        shape: The radargram's shape, delay samples by traces.

    Returns:
        clutter: The simulated power, zero without echo.

    Raises:
        ValueError: When the file is not one array the radargram's size.
    """
    if path.stat().st_size != np.prod(shape) * np.dtype(configs.CLUTTER_TYPE).itemsize:
        raise ValueError(f"{path.name} is not one array the radargram's size.")
    return np.memmap(path, dtype=configs.CLUTTER_TYPE, mode="r", shape=shape)
