"""Reading one CRISM observation off disk and cleaning it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from building.common.pds import images
from building.configs import crism as configs
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


def read_detectors(identifier: str) -> dict[str, Detector]:
    """Read every image one observation was downloaded as, keyed by detector.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in, the wavelength file each of its
            labels names included.

    Returns:
        Both detectors and both geometries, each cube ordered by the wavelength
        file its own label was calibrated against.

    Raises:
        FileNotFoundError: When any of the four images, their labels, or a
            wavelength file a label names is missing.
        ValueError: When a label names a band order this cannot read, or the
            wavelength file does not describe the cube beside it.
    """
    detectors = {}
    for name in configs.DETECTORS:
        # The scan itself, then the geometry published beside it.
        scan = configs.NAMING.product(identifier, configs.OBSERVATION, detector=name)
        cube, label = images.load_cube(configs.CACHE.files(identifier, scan)[".img"])
        geometry = configs.NAMING.product(identifier, configs.GEOMETRY, detector=name)
        planes, geometry_label = images.load_cube(
            configs.CACHE.files(identifier, geometry, configs.GEOMETRY)[".img"]
        )
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
        # Pair each detector's own cube with the geometry beside it.
        detectors[name] = Detector(name, cube, label, table, planes, geometry_label)
    return detectors


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
        cube, mask = masking.bad_pixels(
            detector.cube, detector.wavelengths, detector.name
        )
        cube, mask = atmospheric.remove_atmospheric_bands(
            cube, mask, detector.wavelengths, detector.name
        )
        cube, mask = destripe.remove_spike_columns(
            cube, mask, detector.wavelengths, detector.name
        )
        cube = ratio.ratio_colmed(cube, mask.pixels)
        # Despike only the bands still in play, so the filled ones cannot pull
        # the moving median around at their edges.
        kept = ~mask.bands
        block = np.ascontiguousarray(cube[:, :, kept])
        despike.remove_spikes(
            block, bands_calibration.centres(detector.wavelengths)[kept]
        )
        cube[:, :, kept] = block
        cleaned[name] = replace(detector, cube=cube, mask=mask)
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
    return merge.merge_detectors(identifier, clean_detectors(identifier))
