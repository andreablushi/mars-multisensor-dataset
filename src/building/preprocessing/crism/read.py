"""Reading one CRISM observation off disk and handing it to be cleaned."""

from __future__ import annotations

from building.configs import crism as configs
from building.preprocessing.crism import clean
from building.preprocessing.crism.correction import centre_wavelengths
from building.preprocessing.crism.models.observation import CrismObservation
from common.pds import images, labels


def read_observation(
    identifier: str, spikes: dict[str, float]
) -> CrismObservation | None:
    """Read one observation off disk and have it cleaned and joined onto the grid.

    Args:
        identifier: The observation, its files already in the download cache.
        spikes: The passes, window and sigma the spikes are removed with.

    Returns:
        observation: The observation on the survey's shared grid, or None where no
            detector measured a pixel.

    Raises:
        FileNotFoundError: When no detector of the observation is in the cache.
    """
    scans = {}
    held = []
    for name in configs.Detector:
        image = configs.CACHE.product_files(
            identifier, configs.Kind.OBSERVATION, detector=name
        )[".img"]
        if image.exists():
            cube, label = images.load_cube(image)
            scans[name] = (cube, centre_wavelengths.detector_rows(image, label))
            held.append(label)
    if not scans:
        raise FileNotFoundError(f"No detector of {identifier} is in the cache.")
    geometry, geometry_label = images.load_cube(
        configs.CACHE.product_files(
            identifier, configs.Kind.GEOMETRY, detector=next(iter(scans))
        )[".img"]
    )
    label = labels.merge(*held, geometry_label)
    return clean.clean_observation(identifier, scans, geometry, label, spikes)
