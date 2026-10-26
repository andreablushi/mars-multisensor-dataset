"""Reading one SHARAD track off disk, its geometry joined onto the radargram."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from building.configs import sharad as configs
from building.preprocessing.sharad.models.gate import Gate
from building.preprocessing.sharad.models.observation import SharadObservation
from building.preprocessing.sharad.normalize import clutter_db, radargram_db
from common.pds import images, labels, tables


def read_observation(
    identifier: str,
    looks: int,
    false_alarm: float,
    guard_rows: int,
    echo_margin_db: float,
) -> SharadObservation:
    """Read one radargram and join it to the geometry it was measured at.

    Args:
        identifier: The observation, its files already in the download cache.
        looks: How many traces each delay row of its crops is averaged over.
        false_alarm: The share of a crop's pure noise left over its threshold.
        guard_rows: The rows kept between an echo and the noise beside it.
        echo_margin_db: How far a row's mean power stands over the noise above the
            surface to hold an echo.

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
        Gate(looks, false_alarm, guard_rows, echo_margin_db),
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
