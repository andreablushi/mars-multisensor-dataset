"""Reading one CTX scan off disk and cutting it to the feature it was kept for."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tifffile

from building.common.pds import labels
from building.configs import ctx as configs
from building.preprocessing.common.crop import marked, overlap, polar_overlap, taken
from building.preprocessing.ctx import projection
from building.preprocessing.ctx.models.observation import CtxObservation
from building.preprocessing.ctx.models.sample import BLANK, CtxSample
from shared.models.feature import Feature

# Nothing here reads ASU's no-data tag: what a scan left blank is `BLANK`
logging.getLogger("tifffile").setLevel(logging.ERROR)

# What one build holds per byte of scan; the two offset planes are eight of it
HELD_PER_BYTE = 10

# What the reader, the label and the grids cost whatever size the scan is.
HELD_FLOOR = 256 * 1024**2


def held_bytes(identifier: str) -> int:
    """Return how much memory one build of this scan holds at its peak.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        held: How many bytes to hold for it, floor included.

    Raises:
        FileNotFoundError: When the image is missing.
        tifffile.TiffFileError: When it is not a TIFF this can read.
    """
    # Read off the scan that landed, a projected one running to gigabytes
    files = configs.CACHE.files(identifier, identifier)
    with tifffile.TiffFile(files[configs.SUFFIXES[configs.IMAGE]]) as scan:
        held = scan.pages[0].nbytes
    return HELD_FLOOR + HELD_PER_BYTE * held


def read_observation(identifier: str) -> CtxObservation:
    """Read one scan and place it on the grid its label projects it onto.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        observation: The observation, its image on that grid.

    Raises:
        FileNotFoundError: When the image or its label is missing.
        ValueError: When the label names a projection this cannot read, or the
            image holds more than one plane.
    """
    files = configs.CACHE.files(identifier, identifier)
    label = labels.load(files[configs.SUFFIXES[configs.LABEL]])
    # ASU publishes the pixels as a TIFF rather than beside a label of their own.
    image = files[configs.SUFFIXES[configs.IMAGE]]
    with tifffile.TiffFile(image) as scan:
        held = scan.pages[0].shape
    if len(held) != 2:
        raise ValueError(f"{identifier} holds a {len(held)} dimensional image.")
    return CtxObservation(
        identifier, labels.merge(label), image, *projection.grid_axes(label)
    )


def windowed(image: Path, bounds: tuple[np.ndarray, ...]) -> np.ndarray:
    """Return the pixels one cut keeps, reading no more of the scan than holds them.

    Args:
        image: The TIFF the scan was published as, tiled, so a window of it
            costs the tiles it covers and not the whole file.
        bounds: The lines to keep and then the samples, as the cut left them.

    Returns:
        pixels: The pixels of those lines and samples, as lines by samples.
    """
    lines, samples = bounds
    # A box over the meridian keeps two ends of a strip, so the window spans both.
    top, left = int(lines.min()), int(samples.min())
    window = tifffile.imread(
        image,
        selection=(
            slice(top, int(lines.max()) + 1),
            slice(left, int(samples.max()) + 1),
        ),
    )
    return taken(window, (lines - top, samples - left))


def crop(observation: CtxObservation, frame: Feature) -> CtxSample | None:
    """Return one scan holding only the pixels its feature's box keeps.

    Args:
        observation: The scan as it was read off disk.
        frame: The local frame of the feature it was kept for.

    Returns:
        sample: The scan cut to that feature, or None where it reaches none of it.
    """
    held = (
        polar_overlap(observation.down, observation.across, observation.polar, frame)
        if observation.polar
        else overlap(observation.down, observation.across, observation.separable, frame)
    )
    if held is None:
        return None
    image = windowed(observation.image, held.bounds)
    return CtxSample(
        identifier=observation.identifier,
        position=held.position,
        label=observation.label,
        inside=held.inside,
        valid=marked(image != BLANK),
        image=image,
    )
