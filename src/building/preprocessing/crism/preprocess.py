"""Reading one CRISM observation off disk, cleaning it, and cutting it to a tile."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from building.configs import crism as configs
from building.preprocessing.common import cut, geometry
from building.preprocessing.common.models.samples import Samples
from building.preprocessing.crism.correction import (
    atmospheric,
    bands_calibration,
    despike,
    destripe,
    masking,
    merge,
    ratio,
)
from building.preprocessing.crism.models.detector_cube import DetectorCube
from building.preprocessing.crism.models.observation import (
    ACQUISITION_PLANES,
    CrismObservation,
)
from building.preprocessing.crism.models.sample import CrismSample
from common.models.tile import Tile
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
        detector: Which detector's half of it, `l` or `s`.
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
        detectors: The detectors whose observation and geometry both landed.

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


def placing_detector(identifier: str) -> configs.Detector:
    """Read which detector's geometry places one observation.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        detector: The first detector that landed, in placing order.

    Raises:
        FileNotFoundError: When neither detector landed whole.
    """
    return cached_detectors(identifier)[0]


def read_wavelengths(record: Path) -> np.ndarray:
    """Read the centre wavelength of every column and band one record holds.

    Args:
        record: The `.img` of a wavelength file, which holds a single line.

    Returns:
        wavelengths: The centre wavelength in nm as columns by bands.
    """
    written = images.load_cube(record)[0][0]
    return np.where(written >= UNCALIBRATED, np.nan, written.astype("f8"))


def read_detectors(identifier: str) -> dict[configs.Detector, DetectorCube]:
    """Read every image one observation was downloaded as, keyed by detector.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        detectors: Every detector that landed, each cube in wavelength order.

    Raises:
        FileNotFoundError: When no detector landed or a wavelength file is missing.
        ValueError: When the band order or wavelength file cannot be read.
    """
    detectors = {}
    for name in cached_detectors(identifier):
        cube, label = images.load_cube(
            product_files(identifier, name, configs.Kind.OBSERVATION)[".img"]
        )
        # The wavelength file this half was calibrated against, and no other.
        wavelength = Path(label[configs.WAVELENGTH_KEY]).stem.lower()
        record = configs.CACHE.files(configs.WAVELENGTH_DIR, wavelength)[".img"]
        # Order the bands by wavelength and mark what was never calibrated.
        cube, table = bands_calibration.calibrate(cube, read_wavelengths(record))
        detectors[name] = DetectorCube(name, cube, table)
    return detectors


def read_label(identifier: str) -> dict[str, str]:
    """Read what every product one observation is published as says about it.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        label: Their labels merged, without the calibration software's tuning.

    Raises:
        FileNotFoundError: When neither detector landed, or a label is missing.
    """
    held = [
        labels.load(product_files(identifier, name, configs.Kind.OBSERVATION)[".lbl"])
        for name in cached_detectors(identifier)
    ]
    placing = placing_detector(identifier)
    held.append(
        labels.load(product_files(identifier, placing, configs.Kind.GEOMETRY)[".lbl"])
    )
    merged = labels.merge(*held)
    return {
        key: value
        for key, value in merged.items()
        if not key.startswith(GROUND_SOFTWARE)
    }


def read_geometry(identifier: str) -> np.ndarray:
    """Read the backplanes that place every pixel of one observation.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        backplanes: The placing detector's backplanes, lines by samples by 14.

    Raises:
        FileNotFoundError: When the geometry or its label is missing.
        KeyError: When the label names a sample type this cannot read.
    """
    placing = placing_detector(identifier)
    return images.load_cube(
        product_files(identifier, placing, configs.Kind.GEOMETRY)[".img"]
    )[0]


def clean_detectors(identifier: str) -> dict[configs.Detector, DetectorCube]:
    """Read one observation and refuse everything in it that is not measured.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        detectors: Every detector that landed and measured, filled and with its mask.

    Raises:
        FileNotFoundError: When any file the observation needs is missing.
        ValueError: When a window keeps no band of a cube, or no detector measured.
    """

    def cleaned(detector: DetectorCube) -> DetectorCube:
        """Refuse everything one detector holds that is not measured.

        Args:
            detector: The detector as read, which every step works on in place.

        Returns:
            detector: The same detector, carrying the mask every step left.

        Raises:
            ValueError: When a window keeps no band of the cube.
        """
        cube, table, name = detector.cube, detector.wavelengths, detector.name
        mask = masking.bad_pixels(cube, table, name)
        mask = atmospheric.remove_atmospheric_bands(cube, mask, table, name)
        mask = destripe.remove_spike_columns(cube, mask, table, name)
        mask = ratio.ratio_colmed(cube, mask)
        # Despike only the bands in play, so filled ones cannot pull the median about.
        kept = ~mask.bands
        block = np.ascontiguousarray(cube[:, :, kept])
        despike.remove_spikes(
            block, bands_calibration.centres(table)[kept], mask.pixels
        )
        cube[:, :, kept] = block
        return replace(detector, mask=mask)

    detectors = {}
    for name, detector in read_detectors(identifier).items():
        try:
            detectors[name] = cleaned(detector)
        except masking.NoMeasurement:
            continue
    if not detectors:
        raise ValueError(f"No detector of {identifier} holds a measurement.")
    return detectors


def read_observation(identifier: str) -> CrismObservation:
    """Read one observation, clean it, and join the detectors it has onto the grid.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        observation: The observation on the survey's shared grid.

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


def crop(observation: CrismObservation, frame: Tile) -> CrismSample | None:
    """Return one observation holding only the pixels its tile's box keeps.

    Args:
        observation: The observation with its two detectors joined.
        frame: The local frame of the tile it was kept for.

    Returns:
        sample: The observation cut to that tile, or None where it misses.
    """
    held = cut.overlap(
        Samples(
            observation.latitude, observation.longitude, observation.separable, None
        ),
        frame,
    )
    if held is None:
        return None
    return CrismSample(
        identifier=observation.identifier,
        position=held.position,
        label=observation.label,
        inside=held.inside,
        valid=geometry.marked(geometry.taken(observation.valid, held.bounds)),
        cube=geometry.taken(observation.cube, held.bounds),
        measured_bands=observation.measured_bands,
        **{
            name: geometry.taken(observation.plane(at), held.bounds)
            for name, at in ACQUISITION_PLANES.items()
        },
    )
