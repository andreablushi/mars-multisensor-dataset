"""Writing one crop into the store, as arrays that say what their own axes are."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

import utils.disk.paths as paths
from building.common.layout import Layout
from building.metadata.models.feature import FeatureFrame
from building.preprocessing.common.models.crop import Crop
from utils.disk.slugify import slugify

# What the arrays placing a crop are called, and what the mask beside them is.
NORTH = "north"
EAST = "east"
INSIDE = "inside"

# How a variable names the arrays that place it, which is what a reader turns
# into coordinates rather than into more data.
COORDINATES = "coordinates"

# The most values to hold in one chunk, so a reader takes a patch of a large
# raster without ever reading the whole of it.
CHUNK = 1 << 22

Arrays = dict[str, tuple[np.ndarray, tuple[str, ...]]]


def crop_path(
    frame: FeatureFrame, instrument: str, identifier: str, root: Path
) -> Path:
    """Return where one crop's arrays belong.

    Args:
        frame: The feature the crop was cut to.
        instrument: The instrument that took it, as ODE names it.
        identifier: What that instrument was asked for.
        root: The dataset's own root directory.

    Returns:
        The directory the crop is written in, which need not exist.
    """
    return (
        root
        / slugify(frame.feature_class)
        / slugify(frame.feature_name)
        / slugify(instrument)
        / f"{slugify(identifier)}{paths.CROP_SUFFIX}"
    )


def write_crop(
    held: Crop,
    arrays: Arrays,
    layout: Layout,
    frame: FeatureFrame,
    root: Path = paths.DATASET_ROOT,
) -> Path:
    """Write one crop's arrays down, each saying which axes it runs along.

    Args:
        held: The crop, whose position and mask are written beside the values.
        arrays: What the instrument publishes, keyed by the name to write it as,
            each with the names of its own axes.
        layout: How that instrument's arrays are laid out.
        frame: The feature the crop was cut to.
        root: The dataset's own root directory.

    Returns:
        The directory the crop was written in.
    """
    ground = layout.ground
    # A separable position holds one ground axis each, and any other a value
    # for every sample, so it runs along the whole of the ground.
    north, east = (
        (ground[:1], ground[1:]) if held.position.separable else (ground, ground)
    )
    placed: Arrays = {
        NORTH: (held.position.north, north),
        EAST: (held.position.east, east),
    }
    if held.inside is not None:
        placed[INSIDE] = (held.inside, ground)

    identifier = held.sample.identifier
    path = crop_path(frame, layout.instrument, identifier, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(store=path, mode="w")
    for name, (values, along) in placed.items():
        _written(group, name, np.asarray(values), along)
    for name, (values, along) in arrays.items():
        stored = _written(group, name, np.asarray(values), along)
        stored.attrs[COORDINATES] = f"{NORTH} {EAST}"
    group.attrs.update(
        {
            "instrument": layout.instrument,
            "identifier": identifier,
            "feature_class": frame.feature_class,
            "feature_name": frame.feature_name,
            "measurement": layout.measurement,
            "separable": held.position.separable,
            "centre_lon": frame.centre_lon,
            "centre_lat": frame.centre_lat,
        }
    )
    return path


def _written(
    group: zarr.Group, name: str, values: np.ndarray, dims: tuple[str, ...]
) -> zarr.Array:
    """Write one array into the group, chunked so a patch is read on its own.

    Args:
        group: The crop's group.
        name: What to call the array.
        values: What to write.
        dims: What each of its axes is called.

    Returns:
        The array that was written.
    """
    stored = group.create_array(
        name,
        shape=values.shape,
        dtype=values.dtype,
        chunks=_chunks(values.shape, values.dtype.itemsize),
        dimension_names=dims,
    )
    stored[...] = values
    return stored


def _chunks(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    """Return a chunk shape holding no more values than one chunk should.

    Args:
        shape: The array's shape.
        itemsize: How many bytes one of its values takes.

    Returns:
        The chunk shape, the trailing axes kept whole and the leading ones cut
        down until a chunk fits, so a spectrum or a trace is never split.
    """
    chunk = list(shape)
    held = max(1, CHUNK // max(itemsize, 1))
    for axis in range(len(shape)):
        if int(np.prod(chunk)) <= held:
            break
        rest = int(np.prod(chunk[axis + 1 :])) or 1
        chunk[axis] = max(1, min(chunk[axis], held // rest))
    return tuple(chunk)
