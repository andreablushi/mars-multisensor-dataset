"""Writing one crop into the store, as arrays that say what their own axes are."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.building import paths
from common.building.common.layout import GROUND, Layout
from common.building.preprocessing.common import relative_positioning
from common.building.preprocessing.common.models.sample import Sample
from common.disk.files import atomic_path
from common.disk.slugify import slugify
from common.maths import physics
from common.models.tile import Tile

# What the arrays placing a crop are called, and what the masks beside them are.
NORTH = "north"
EAST = "east"
INSIDE = "inside"
VALID = "valid"
MEASURED = "measured_ground"

# What the placing arrays are measured in, degrees from the centre or a grid's metres.
DEGREES = "degrees"
METRES = "metres"

# What the crop is described by: its axes, its tile, its units and its label.
META = "meta"


def sample_path(frame: Tile, instrument: str, identifier: str, root: Path) -> Path:
    """Return where one cropped observation's arrays belong.

    Args:
        frame: The tile it was cut to.
        instrument: The instrument that took it, as ODE names it.
        identifier: What that instrument was asked for.
        root: The dataset's own root directory.

    Returns:
        path: The file it is written as, which need not exist.
    """
    return (
        root
        / frame.band_name
        / frame.column_name
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
    frame: Tile,
    root: Path,
) -> Path:
    """Write one sample down, its arrays and what describes them in one file.

    Args:
        held: The sample, whose position and masks are written beside the values.
        layout: How that instrument's arrays are laid out, which names every
            one of them the sample is read for.
        frame: The tile it was cut to.
        root: The dataset's own root directory.

    Returns:
        path: The file it was written as.

    Raises:
        ValueError: When the layout declares an array beside the measurement that
            the crop itself carries none of.
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
        MEASURED: ground,
        **layout.beside,
    }
    arrays = {}
    for name in layout.beside:
        alongside = getattr(held, name)
        if alongside is None:
            raise ValueError(f"{layout.instrument} declares {name} but holds none.")
        arrays[name] = native(alongside)
    values = native(getattr(held, layout.measurement))
    arrays[layout.measurement] = values.astype(
        layout.stored or values.dtype, copy=False
    )
    arrays[NORTH] = native(held.position.north).astype(
        relative_positioning.STORED, copy=False
    )
    arrays[EAST] = native(held.position.east).astype(
        relative_positioning.STORED, copy=False
    )
    arrays[MEASURED] = native(held.measured_ground)
    for name, mask in ((INSIDE, held.inside), (VALID, held.valid)):
        # The two the rooted mask is made of, kept for whoever wants them apart.
        if mask is not None:
            arrays[name] = native(mask)
            along[name] = ground

    path = sample_path(frame, layout.instrument, held.identifier, root)
    grid = held.position.polar
    described = {
        "instrument": layout.instrument,
        "identifier": held.identifier,
        "tile": frame.name,
        "band": frame.band,
        "column": frame.column,
        "measurement": layout.measurement,
        "separable": held.position.separable,
        "centre_lon": frame.centre_lon,
        "centre_lat": frame.centre_lat,
        "box": {
            "min_lat": frame.min_lat,
            "max_lat": frame.max_lat,
            "west_lon": frame.west_lon,
            "east_lon": frame.east_lon,
        },
        "position_units": DEGREES if grid is None else METRES,
        "radii_m": [physics.EQUATORIAL_RADIUS_M, physics.POLAR_RADIUS_M],
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
