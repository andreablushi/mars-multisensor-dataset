"""Cleaning each CRISM detector step by step, then joining, lighting and levelling."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from mars_multisensor_dataset.building.configs import crism as configs
from mars_multisensor_dataset.building.preprocessing.crism import merge
from mars_multisensor_dataset.building.preprocessing.crism.correction import (
    atmosphere,
    centre_wavelengths,
    mask,
    normalize,
    photometric,
    remove_outliers,
)
from mars_multisensor_dataset.building.preprocessing.crism.models.observation import (
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
    levelling: int,
    join_ratio: tuple[float, float],
) -> CrismObservation | None:
    """Clean every detector of one observation and join them onto the survey's grid.

    Args:
        identifier: The observation, which names it where a detector is dropped.
        scans: Each detector's raw cube and the detector row of every band.
        geometry: The backplanes that place every pixel.
        label: What every product the observation was published as says of it.
        spikes: The passes, window and sigma the spikes are removed with.
        levelling: How many lines a column is levelled over along the track.
        join_ratio: The lowest and highest ratio of the first infrared band to the
            last visible one an observation keeps.

    Returns:
        observation: The joined observation, or None where a detector measured
            nothing or the two do not meet at their join.
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
    if dropped := [name for name in configs.Detector if name not in cleaned]:
        missed = ", ".join(dropped)
        print(f"note {identifier} [CRISM]: no pixel measured on {missed}", flush=True)
        return None
    observation = merge.merge_detectors(cleaned, geometry, label)
    valid = photometric.photometric_valid(
        observation.cube,
        observation.geometry[:, :, ACQUISITION_PLANES["incidence_deg"]],
        observation.valid,
        observation.measured_bands,
    )
    remove_outliers.level_columns(observation.cube, valid, levelling)
    cube = normalize.normalized_spectra(observation.cube, valid)
    visible = centre_wavelengths.DETECTOR_SLOTS[configs.Detector.VISIBLE][-1]
    infrared = centre_wavelengths.DETECTOR_SLOTS[configs.Detector.INFRARED][0]
    ratio = float(np.nanmedian(cube[valid, infrared] / cube[valid, visible]))
    if not join_ratio[0] <= ratio <= join_ratio[1]:
        print(f"note {identifier} [CRISM]: detectors join at {ratio:.2f}", flush=True)
        return None
    return replace(observation, cube=cube, valid=valid)
