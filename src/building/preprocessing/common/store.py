"""Writing one crop into the store, as arrays that say what their own axes are."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import utils.disk.paths as paths
from building.common.layout import GROUND, Layout
from building.models.feature import FeatureFrame
from building.preprocessing.common.models.sample import Sample
from utils.disk.files import atomic_path
from utils.disk.slugify import slugify
from utils.geometry import geodesy

# What the arrays placing a crop are called, and what the masks beside them are.
NORTH = "north"
EAST = "east"
INSIDE = "inside"
VALID = "valid"

# What the placing arrays are measured in, degrees from the centre or a grid's metres.
DEGREES = "degrees"
METRES = "metres"

# What the crop is described by: its axes, its feature, its units and its label.
META = "meta"


def sample_path(
    frame: FeatureFrame, instrument: str, identifier: str, root: Path
) -> Path:
    """Return where one cropped observation's arrays belong.

    Args:
        frame: The feature it was cut to.
        instrument: The instrument that took it, as ODE names it.
        identifier: What that instrument was asked for.
        root: The dataset's own root directory.

    Returns:
        path: The file it is written as, which need not exist.
    """
    return (
        root
        / slugify(frame.feature_class)
        / slugify(frame.feature_name)
        / slugify(instrument)
        / f"{slugify(identifier)}{paths.SAMPLE_SUFFIX}"
    )


def native(values: np.ndarray) -> np.ndarray:
    """Return one array in the byte order the machine reads.

    Args:
        values: The values to store, which a PDS archive publishes most
            significant byte first whatever the machine reading it is.

    Returns:
        values: The same values in the machine's own order, ready to hand to a tensor.
    """
    held = np.asarray(values)
    return held.astype(held.dtype.newbyteorder("="), copy=False)


def write_sample(
    held: Sample,
    layout: Layout,
    frame: FeatureFrame,
    root: Path,
) -> Path:
    """Write one sample down, its arrays and what describes them in one file.

    Args:
        held: The sample, whose position and masks are written beside the values.
        layout: How that instrument's arrays are laid out, which names every
            one of them the sample is read for.
        frame: The feature it was cut to.
        root: The dataset's own root directory.

    Returns:
        path: The file it was written as.
    """
    ground = tuple(
        name
        for name, holds in zip(layout.dims, layout.axes, strict=True)
        if holds == GROUND
    )
    # A separable position holds one ground axis each, any other a value per sample.
    north, east = held.position.dims_along(ground)
    along = {
        layout.measurement: layout.dims,
        NORTH: north,
        EAST: east,
        **layout.beside,
    }
    arrays = {name: native(getattr(held, name)) for name in layout.beside}
    arrays[layout.measurement] = native(getattr(held, layout.measurement))
    arrays[NORTH] = native(held.position.north)
    arrays[EAST] = native(held.position.east)
    for name, mask in ((INSIDE, held.inside), (VALID, held.valid)):
        # A mask marking every sample was never stored, so it is never read.
        if mask is not None:
            arrays[name] = native(mask)
            along[name] = ground

    path = sample_path(frame, layout.instrument, held.identifier, root)
    grid = held.position.polar
    described = {
        "instrument": layout.instrument,
        "identifier": held.identifier,
        "feature_class": frame.feature_class,
        "feature_name": frame.feature_name,
        "measurement": layout.measurement,
        "separable": held.position.separable,
        "centre_lon": frame.centre_lon,
        "centre_lat": frame.centre_lat,
        "position_units": DEGREES if grid is None else METRES,
        "radii_m": [geodesy.EQUATORIAL_RADIUS_M, geodesy.POLAR_RADIUS_M],
        "polar": None if grid is None else list(grid),
        "dims": {name: list(axes) for name, axes in along.items()},
        "axes": list(layout.axes),
        "ground": list(ground),
        "label": held.label,
    }
    # Compressed, and written whole then moved, so a crop a reader finds was finished.
    with atomic_path(path) as tmp, tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays, **{META: np.array(json.dumps(described))})
    return path
