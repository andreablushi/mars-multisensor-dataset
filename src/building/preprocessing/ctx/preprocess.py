"""Reading one CTX scan off disk and cutting it to the feature it was kept for."""

from __future__ import annotations

import numpy as np
import tifffile

from building.common.pds import labels
from building.configs import ctx as configs
from building.models.feature import FeatureFrame
from building.preprocessing.common.crop import TURN, marked, overlap, taken
from building.preprocessing.common.models.overlap import Overlap
from building.preprocessing.common.models.relative_position import RelativePosition
from building.preprocessing.ctx import projection
from building.preprocessing.ctx.models.observation import CtxObservation
from building.preprocessing.ctx.models.sample import BLANK, CtxSample
from utils.geometry import geodesy

# The longest segment the feature's box is walked in before it is projected, in
# degrees. A chord this short leaves the arc it stands in by under a pixel.
STEP = 0.1

# How many pixels of a polar cut are turned back into degrees at once. A feature
# can want most of a scan, so the mask over it is built a block of lines a time.
BLOCK = 4_000_000


def read_observation(identifier: str) -> CtxObservation:
    """Read one scan and place it on the grid its label projects it onto.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        The observation, its image on that grid.

    Raises:
        FileNotFoundError: When the image or its label is missing.
        ValueError: When the label names a projection this cannot read, or the
            image holds more than one plane.
    """
    files = configs.CACHE.files(identifier, identifier)
    label = labels.load(files[configs.SUFFIXES[configs.LABEL]])
    # ASU publishes the pixels as a TIFF rather than beside a label of their own.
    image = tifffile.imread(files[configs.SUFFIXES[configs.IMAGE]])
    if image.ndim != 2:
        raise ValueError(f"{identifier} holds a {image.ndim} dimensional image.")
    return CtxObservation(
        identifier, labels.merge(label), image, *projection.load(label)
    )


def crop(observation: CtxObservation, frame: FeatureFrame) -> CtxSample | None:
    """Return one scan holding only the pixels its feature's box keeps.

    Args:
        observation: The scan as it was read off disk.
        frame: The local frame of the feature it was kept for.

    Returns:
        The scan cut to that feature, or None where it reaches none of it.
    """
    held = (
        polar_overlap(observation, frame)
        if observation.polar
        else overlap(observation, frame)
    )
    if held is None:
        return None
    image = taken(observation.image, held.bounds)
    return CtxSample(
        identifier=observation.identifier,
        position=held.position,
        label=observation.label,
        inside=held.inside,
        valid=marked(image != BLANK),
        image=image,
    )


def polar_overlap(observation: CtxObservation, frame: FeatureFrame) -> Overlap | None:
    """Return what one feature's box keeps of one scan placed on a polar grid.

    The box is walked and projected rather than the grid being turned back into
    degrees, so a scan of a thousand million pixels is never held as degrees to
    find out which handful of them a feature wants.

    Args:
        observation: The scan as it was read off disk, on the polar grid its
            label projects it onto.
        frame: The feature's local frame, carrying the box the catalogue gives
            it.

    Returns:
        What the box keeps, or None where the scan reaches none of it.
    """
    grid = observation.polar
    ring = geodesy.stereographic_forward(
        *geodesy.bbox_ring(
            frame.min_lat, frame.max_lat, frame.west_lon, frame.east_lon, STEP
        ),
        *grid,
    )
    # The box projects to a sector, and the ring its edge traces bounds it.
    lines = np.flatnonzero(
        (observation.down >= ring[1].min()) & (observation.down <= ring[1].max())
    )
    samples = np.flatnonzero(
        (observation.across >= ring[0].min()) & (observation.across <= ring[0].max())
    )
    if not lines.size or not samples.size:
        return None
    # Only the rectangle the sector stands in is crossed back into degrees, and
    # only a block of its lines at a time.
    span = geodesy.longitude_span(frame.west_lon, frame.east_lon)
    across = observation.across[samples][None, :]
    inside = np.empty((lines.size, samples.size), dtype=bool)
    reach = max(1, BLOCK // samples.size)
    for start in range(0, lines.size, reach):
        block = slice(start, start + reach)
        lon, lat = geodesy.stereographic_inverse(
            across, observation.down[lines[block]][:, None], *grid
        )
        inside[block] = (
            (lat >= frame.min_lat)
            & (lat <= frame.max_lat)
            & ((lon - frame.west_lon) % TURN <= span)
        )
    if not inside.any():
        return None
    centre_x, centre_y = geodesy.stereographic_forward(
        frame.centre_lon, frame.centre_lat, *grid
    )
    return Overlap(
        (lines, samples),
        marked(inside),
        RelativePosition(
            observation.down[lines] - float(centre_y),
            observation.across[samples] - float(centre_x),
            True,
            grid,
        ),
    )
