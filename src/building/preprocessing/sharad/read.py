"""Reading one SHARAD track off disk, its geometry joined onto the radargram."""

from __future__ import annotations

import numpy as np

from building.configs import sharad as configs
from building.preprocessing.sharad.models.observation import SharadObservation
from building.preprocessing.sharad.normalize import normalized_power
from common.pds import images, labels, tables


def read_observation(identifier: str) -> SharadObservation:
    """Read one radargram and join it to the geometry it was measured at.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        observation: The traces normalized, with their geometry and clutter
            simulation.

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
    simulated = held[configs.Kind.CLUTTER][".img"]
    if simulated.stat().st_size != power.size * np.dtype(configs.CLUTTER_TYPE).itemsize:
        raise ValueError(f"{simulated.name} is not one array the radargram's size.")
    clutter = np.memmap(
        simulated, dtype=configs.CLUTTER_TYPE, mode="r", shape=power.shape
    )
    return SharadObservation(
        labels.merge(sounding, geometry_label),
        normalized_power(power, clutter),
        clutter,
        geometry,
    )
