"""Writing one crop into the store, as arrays that say what their own axes are."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import utils.disk.paths as paths
from building.common.layout import Layout
from building.models.feature import FeatureFrame
from building.preprocessing.common.models.sample import Sample
from utils.disk.files import atomic_path
from utils.disk.slugify import slugify

# What the arrays placing a crop are called, and what the masks beside them are.
NORTH = "north"
EAST = "east"
INSIDE = "inside"
VALID = "valid"

# What the crop is described by, beside the arrays it holds: what each array's
# own axes are called and which of them are ground, where the feature it was cut
# to sits, and the label every product it was published as carries.
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
        The file it is written as, which need not exist.
    """
    return (
        root
        / slugify(frame.feature_class)
        / slugify(frame.feature_name)
        / slugify(instrument)
        / f"{slugify(identifier)}{paths.SAMPLE_SUFFIX}"
    )


def write_sample(
    held: Sample,
    layout: Layout,
    frame: FeatureFrame,
    root: Path = paths.DATASET_ROOT,
) -> Path:
    """Write one sample down, its arrays and what describes them in one file.

    Args:
        held: The sample, whose position and masks are written beside the values.
        layout: How that instrument's arrays are laid out, which names every
            one of them the sample is read for.
        frame: The feature it was cut to.
        root: The dataset's own root directory.

    Returns:
        The file it was written as.
    """
    ground = layout.ground
    # A separable position holds one ground axis each, and any other a value
    # for every sample, so it runs along the whole of the ground.
    north, east = (
        (ground[:1], ground[1:]) if held.position.separable else (ground, ground)
    )
    along = {
        layout.measurement: layout.dims,
        NORTH: north,
        EAST: east,
        **layout.beside,
    }
    arrays = {name: np.asarray(getattr(held, name)) for name in layout.beside}
    arrays[layout.measurement] = np.asarray(getattr(held, layout.measurement))
    arrays[NORTH] = np.asarray(held.position.north)
    arrays[EAST] = np.asarray(held.position.east)
    for name, mask in ((INSIDE, held.inside), (VALID, held.valid)):
        # A mask marking every sample was never stored, so it is never read.
        if mask is not None:
            arrays[name] = np.asarray(mask)
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
        "dims": {name: list(axes) for name, axes in along.items()},
        "axes": list(layout.axes),
        "ground": list(ground),
        "label": held.label,
    }
    # Written whole and moved into place, so a crop a reader finds is a crop
    # that was finished and never one a run was interrupted partway through.
    with atomic_path(path) as tmp, tmp.open("wb") as handle:
        np.savez(handle, **arrays, **{META: np.array(json.dumps(described))})
    return path
