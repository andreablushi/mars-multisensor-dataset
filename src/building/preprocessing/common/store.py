"""Writing one crop into the store, as arrays that say what their own axes are."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from building import paths
from building.common.layout import GROUND, Layout
from building.preprocessing.common import relative_positioning
from building.preprocessing.common.models.sample import Sample
from shared.disk.files import atomic_path
from shared.disk.slugify import slugify
from shared.maths import physics
from shared.models.feature import Feature

# What the arrays placing a crop are called, and what the masks beside them are.
NORTH = "north"
EAST = "east"
INSIDE = "inside"
VALID = "valid"
MEASURED = "measured"

# The frame the placing arrays hold, ground metres from the feature's own centre.
FRAME = "aeqd_m"

# What the crop is described by: its axes, its feature, its frame and its label.
META = "meta"


def sample_path(frame: Feature, instrument: str, identifier: str, root: Path) -> Path:
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
    frame: Feature,
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
    # The frame is not separable, so both offsets run over every ground axis.
    north, east = relative_positioning.ground_metres(held.position, frame)
    along = {
        layout.measurement: layout.dims,
        NORTH: ground,
        EAST: ground,
        MEASURED: ground,
        **layout.beside,
    }
    arrays = {name: native(getattr(held, name)) for name in layout.beside}
    arrays[layout.measurement] = native(getattr(held, layout.measurement))
    arrays[NORTH] = north
    arrays[EAST] = east
    arrays[MEASURED] = native(held.measured)
    for name, mask in ((INSIDE, held.inside), (VALID, held.valid)):
        # The two the rooted mask is made of, kept for whoever wants them apart.
        if mask is not None:
            arrays[name] = native(mask)
            along[name] = ground

    path = sample_path(frame, layout.instrument, held.identifier, root)
    described = {
        "instrument": layout.instrument,
        "identifier": held.identifier,
        "feature_class": frame.feature_class,
        "feature_name": frame.feature_name,
        "measurement": layout.measurement,
        "separable": held.position.separable,
        "centre_lon": frame.centre_lon,
        "centre_lat": frame.centre_lat,
        "frame": FRAME,
        "radii_m": [physics.EQUATORIAL_RADIUS_M, physics.POLAR_RADIUS_M],
        "dims": {name: list(axes) for name, axes in along.items()},
        "axes": list(layout.axes),
        "ground": list(ground),
        "label": held.label,
    }
    # Compressed, and written whole then moved, so a crop a reader finds was finished.
    with atomic_path(path) as tmp, tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays, **{META: np.array(json.dumps(described))})
    return path
