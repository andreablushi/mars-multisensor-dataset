"""One product fetched, read and cut to a tile, the way a training build does it."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from typing import Any

import httpx
import ipywidgets as widgets
import numpy as np

from analysis import paths
from analysis.selector.artifacts import read_selection
from analysis.visualization import panels
from building import paths as built
from building.common.layout import Axis
from building.metadata.dataset import read_normalization
from building.metadata.observation import observation_metadata
from building.models.instrument import INSTRUMENTS, observation_identifier
from building.preprocessing.common.models.sample import Sample
from building.preprocessing.common.store import sample_path
from common.config import training_settings
from common.disk.files import read_json, write_json
from common.disk.slugify import slugify
from common.fetch.http import TLS_CONTEXT
from common.models.tile import Tile

CLIP = 2.0
STD_FLOOR = 1e-6


def fetch_product(name: str, identifier: str, frame: Tile) -> None:
    """Bring one product into the build's cache and place it, unless it is there.

    Args:
        name: The instrument, as the build names it.
        identifier: What that instrument is asked for, its observation.
        frame: The tile it is fetched for.
    """
    instrument = INSTRUMENTS[name]
    with httpx.Client(verify=TLS_CONTEXT) as client:
        instrument.fetch(identifier, client, (frame,))
    if instrument.place:
        instrument.place(identifier)


def product_observation(name: str, identifier: str) -> Any:
    """Read one fetched product with the settings a training build hands it."""
    preprocessing = training_settings().preprocessing.get(name, {})
    return INSTRUMENTS[name].read_observation(identifier, **preprocessing)


def tile_sample(name: str, observation: Any, frame: Tile) -> Sample:
    """Cut a read product to a tile, as a training build does.

    Args:
        name: The instrument, as the build names it.
        observation: The product as its reader handed it back.
        frame: The tile it is cut to.

    Returns:
        sample: The crop the build would write.

    Raises:
        ValueError: When the product measured nothing or reaches none of the tile.
    """
    sample = None if observation is None else INSTRUMENTS[name].crop(observation, frame)
    if sample is None:
        raise ValueError(f"{name} holds nothing of tile {frame.name}.")
    return sample


def selected_start(name: str, identifier: str, frame: Tile) -> datetime | None:
    """Return when the selection says one product kept for a tile was taken.

    Args:
        name: The instrument, as the build names it.
        identifier: What that instrument was asked for, its observation.
        frame: The tile it was kept for.

    Returns:
        started: Its start time, or None where the selection keeps no such product.
    """
    return next(
        (
            kept.t_start
            for one in read_selection()
            if one.tile.tile == frame.name
            for kept in one.observations
            if kept.iid == name
            and observation_identifier(kept.iid, kept.pdsid) == identifier
        ),
        None,
    )


def acquisition_table(
    name: str, identifier: str, sample: Sample, frame: Tile
) -> widgets.HTML:
    """Tabulate what one built crop was taken under, as the build's index records it."""
    layout = INSTRUMENTS[name].layout
    path = str(sample_path(frame, name, identifier))
    record = observation_metadata(sample, frame, layout, identifier, path, None)
    if record.t_start is None:
        started = selected_start(name, identifier, frame)
        record = observation_metadata(sample, frame, layout, identifier, path, started)
    rows = [
        ("Product", identifier),
        ("Tile", frame.name),
        ("Started", str(record.t_start)),
        ("Ended", str(record.t_end)),
        *(
            (field, f"{value:.4g}")
            for field, value in asdict(record.acquisition).items()
            if value is not None
        ),
    ]
    return panels.written("Acquisition", panels.STATISTIC_VALUE, rows)


def kept_panel(
    name: str,
    identifier: str,
    frame: Tile,
    render: Callable[[str, Tile], widgets.VBox],
) -> widgets.VBox:
    """Return one product's panel over a tile, drawn once and then read off disk.

    Args:
        name: The instrument, as the build names it.
        identifier: What that instrument was asked for, its observation.
        frame: The tile it is drawn over.
        render: What draws the panel, handed the identifier and the tile.

    Returns:
        panel: Its images and tables, in the order they were drawn.
    """
    path = (
        paths.PANELS_ROOT / slugify(name) / f"{slugify(identifier)}_{frame.name}.json"
    )
    if not path.exists():
        write_json(
            path,
            [
                {"png": base64.b64encode(child.value).decode()}
                if isinstance(child, widgets.Image)
                else {"html": child.value}
                for child in render(identifier, frame).children
            ],
        )
    return widgets.VBox(
        [
            panels.png_widget(base64.b64decode(child["png"]))
            if "png" in child
            else widgets.HTML(child["html"])
            for child in read_json(path)
        ]
    )


def model_input(name: str, sample: Sample) -> np.ndarray:
    """Return a built crop's values as the MAE loader hands them to the model.

    Args:
        name: The instrument, as the build names it.
        sample: The crop the build would write.

    Returns:
        values: Standardised by the training build's constants, CRISM clipped.
    """
    layout = INSTRUMENTS[name].layout
    values = getattr(sample, layout.measurement)
    values = values.astype(layout.stored or values.dtype)
    constants = read_normalization(built.dataset_root(training_settings().name))[name]
    # Mirrors standardised_values and read_observation in mars-multisensor-mae's store
    shape = [-1 if axis == Axis.WAVELENGTH else 1 for axis in layout.axes]
    mean, std = (
        np.asarray(constants[key], np.float32).reshape(shape) for key in ("mean", "std")
    )
    standardised = (values - mean) / np.maximum(std, STD_FLOOR)
    if name == "CRISM":
        measured = sample.measured_ground[..., None] & np.isfinite(standardised)
        standardised = np.where(measured, np.clip(standardised, -CLIP, CLIP), 0.0)
    return standardised


def scaled_input(values: np.ndarray) -> np.ndarray:
    """Return model input mapped from -2 to 2 onto 0 to 1, to be drawn."""
    return np.clip((values + CLIP) / (2 * CLIP), 0.0, 1.0)
