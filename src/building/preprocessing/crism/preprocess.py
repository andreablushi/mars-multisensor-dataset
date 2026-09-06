"""Reading one CRISM observation off disk, cleaning it, and cutting it to a feature."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from building.common.pds import images, labels
from building.configs import crism as configs
from building.models.feature import FeatureFrame
from building.preprocessing.common.crop import marked, overlap, taken
from building.preprocessing.crism import configs as cleaning
from building.preprocessing.crism.correction import (
    atmospheric,
    bands_calibration,
    despike,
    destripe,
    masking,
    merge,
    ratio,
)
from building.preprocessing.crism.models.detector import Detector
from building.preprocessing.crism.models.observation import CrismObservation
from building.preprocessing.crism.models.sample import CrismSample


def read_detectors(identifier: str) -> dict[str, Detector]:
    """Read every image one observation was downloaded as, keyed by detector.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in, the wavelength file each of its
            labels names included.

    Returns:
        Both detectors, each cube ordered by the wavelength file its own label
        was calibrated against.

    Raises:
        FileNotFoundError: When either image, its label, or a wavelength file
            a label names is missing.
        ValueError: When a label names a band order this cannot read, or the
            wavelength file does not describe the cube beside it.
    """
    detectors = {}
    for name in configs.DETECTORS:
        scan = configs.NAMING.product(identifier, configs.OBSERVATION, detector=name)
        cube, label = images.load_cube(configs.CACHE.files(identifier, scan)[".img"])
        # The wavelength file this half was calibrated against, and no other.
        wavelength = Path(label[configs.WAVELENGTH_KEY]).stem.lower()
        record = configs.CACHE.files(configs.WAVELENGTH_DIR, wavelength)[".img"]
        # A wavelength file holds one line, so its cube is one grid deep.
        written = images.load_cube(record)[0][0]
        # Say what was never calibrated with NaN rather than a number.
        wavelengths = np.where(
            written >= cleaning.UNCALIBRATED, np.nan, written.astype("f8")
        )
        # Order the bands by wavelength and mark what was never calibrated.
        cube, table = bands_calibration.calibrate(cube, wavelengths)
        detectors[name] = Detector(name, cube, table)
    return detectors


def read_label(identifier: str) -> dict[str, str]:
    """Read what every product one observation is published as says about it.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        Their labels merged into one, without the tuning of the software that
        calibrated them.

    Raises:
        FileNotFoundError: When a label is missing.
    """
    held = []
    for name in configs.DETECTORS:
        scan = configs.NAMING.product(identifier, configs.OBSERVATION, detector=name)
        held.append(labels.load(configs.CACHE.files(identifier, scan)[".lbl"]))
    product = configs.NAMING.product(
        identifier, configs.GEOMETRY, detector=cleaning.PLACING_DETECTOR
    )
    held.append(
        labels.load(configs.CACHE.files(identifier, product, configs.GEOMETRY)[".lbl"])
    )
    merged = labels.merge(*held)
    return {
        key: value
        for key, value in merged.items()
        if not key.startswith(cleaning.GROUND_SOFTWARE)
    }


def read_geometry(identifier: str) -> np.ndarray:
    """Read the backplanes that place every pixel of one observation.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        The backplanes of the detector that places it, as lines by samples by
        fourteen.

    Raises:
        FileNotFoundError: When the geometry or its label is missing.
        KeyError: When the label names a sample type this cannot read.
    """
    product = configs.NAMING.product(
        identifier, configs.GEOMETRY, detector=cleaning.PLACING_DETECTOR
    )
    return images.load_cube(
        configs.CACHE.files(identifier, product, configs.GEOMETRY)[".img"]
    )[0]


def clean_detectors(identifier: str) -> dict[str, Detector]:
    """Read one observation and refuse everything in it that is not measured.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        Both detectors, each cube filled where it was not measured and its mask
        set beside it.

    Raises:
        FileNotFoundError: When any file the observation needs is missing.
        ValueError: When a window keeps no band of a cube.
    """
    cleaned = {}
    for name, detector in read_detectors(identifier).items():
        # Every step works on the one cube, so the chain holds no second copy.
        cube, table = detector.cube, detector.wavelengths
        mask = masking.bad_pixels(cube, table, name)
        mask = atmospheric.remove_atmospheric_bands(cube, mask, table, name)
        mask = destripe.remove_spike_columns(cube, mask, table, name)
        ratio.ratio_colmed(cube, mask.pixels)
        # Despike only the bands in play, so filled ones cannot pull the median about.
        kept = ~mask.bands
        block = np.ascontiguousarray(cube[:, :, kept])
        despike.remove_spikes(block, bands_calibration.centres(table)[kept])
        cube[:, :, kept] = block
        cleaned[name] = replace(detector, mask=mask)
    return cleaned


def read_observation(identifier: str) -> CrismObservation:
    """Read one observation, clean it, and join its two detectors into one cube.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        The observation, its bands ascending in wavelength.

    Raises:
        FileNotFoundError: When any file the observation needs is missing.
        ValueError: When a window keeps no band of a cube.
    """
    return merge.merge_detectors(
        identifier,
        clean_detectors(identifier),
        read_geometry(identifier),
        read_label(identifier),
    )


def crop(observation: CrismObservation, frame: FeatureFrame) -> CrismSample | None:
    """Return one observation holding only the pixels its feature's box keeps.

    Args:
        observation: The observation with its two detectors joined.
        frame: The local frame of the feature it was kept for.

    Returns:
        The observation cut to that feature, its bands left whole, or None
        where it reaches none of it.
    """
    held = overlap(observation, frame)
    if held is None:
        return None
    # Calibrated column by column, so only that axis cuts and the bands stay whole.
    columns = held.bounds[1]
    return CrismSample(
        identifier=observation.identifier,
        position=held.position,
        label=observation.label,
        inside=held.inside,
        valid=marked(taken(observation.valid, held.bounds)),
        cube=taken(observation.cube, held.bounds),
        wavelengths=observation.wavelengths[columns],
        columns=observation.columns[columns],
    )
