"""Cleaning each CRISM detector step by step, then joining and lighting them."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism import merge
from building.preprocessing.crism.correction import (
    atmosphere,
    centre_wavelengths,
    mask,
    photometric,
    remove_outliers,
)
from building.preprocessing.crism.models.observation import (
    ACQUISITION_PLANES,
    CrismObservation,
    DetectorCube,
)


def clean_observation(
    identifier: str,
    scans: dict[configs.Detector, tuple[np.ndarray, np.ndarray]],
    geometry: np.ndarray,
    label: dict[str, str],
    spikes: dict[str, float],
) -> CrismObservation | None:
    """Clean every detector of one observation and join them onto the survey's grid.

    Args:
        identifier: The observation, which names it where a detector is dropped.
        scans: Each detector's raw cube and the detector row of every band.
        geometry: The backplanes that place every pixel.
        label: What every product the observation was published as says of it.
        spikes: The passes, window and sigma the spikes are removed with.

    Returns:
        observation: The joined observation, or None where no detector measured.
    """
    cleaned = {}
    for name, (cube, rows) in scans.items():
        table = centre_wavelengths.read_calibration(
            configs.WAVELENGTH_FILES[name], rows
        ).astype("f8")
        columns = np.isnan(table).all(axis=1)
        bands = centre_wavelengths.window_bands(table, name)
        table = table[:, bands]
        block = np.ascontiguousarray(cube[:, :, bands], dtype="f4")
        refused = mask.refused_pixels(block, columns)
        if refused.all():
            continue
        if name == configs.Detector.INFRARED:
            transmission = atmosphere.read_transmission(label, rows)[:, bands]
            depth = atmosphere.co2_depth(
                block, refused, transmission, centre_wavelengths.band_centres(table)
            )
            block, table, transmission = atmosphere.remove_co2_bands(
                block, table, transmission
            )
            atmosphere.divide_transmission(block, transmission, depth)
        remove_outliers.flat_field(block, refused)
        remove_outliers.remove_spikes(block, refused, **spikes)
        cleaned[name] = DetectorCube(block, table, columns, refused)
    if dropped := [name for name in scans if name not in cleaned]:
        missed = ", ".join(dropped)
        print(f"note {identifier} [CRISM]: no pixel measured on {missed}", flush=True)
    if not cleaned:
        return None
    observation = merge.merge_detectors(cleaned, geometry, label)
    valid = photometric.photometric_valid(
        observation.cube,
        observation.geometry[:, :, ACQUISITION_PLANES["incidence_deg"]],
        observation.valid,
        observation.measured_bands,
    )
    return replace(observation, valid=valid)
