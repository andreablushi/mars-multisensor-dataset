"""One product read and cut to a tile the way a training build does, then drawn."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import ipywidgets as widgets
import numpy as np
from matplotlib.figure import Figure

from analysis.selector.artifacts import read_selection
from building import paths as built
from building.common.layout import Axis
from building.metadata.dataset import read_normalization
from building.metadata.index import read_observation_metadata
from building.metadata.observation import (
    measured_mask,
    measured_statistics,
    observation_metadata,
)
from building.models.instrument import INSTRUMENTS, observation_identifier
from building.preprocessing.common.models.sample import Sample
from building.preprocessing.common.store import sample_path
from common.config import training_settings
from common.models.tile import Tile
from visualization import panels
from visualization.config import STATISTIC_VALUE
from visualization.training_qualitative import config


def observed_sample(name: str, identifier: str, frame: Tile) -> tuple[Any, Sample]:
    """Read one fetched product and cut it to a tile, as a training build does.

    Args:
        name: The instrument, as the build names it.
        identifier: What that instrument was asked for, its observation.
        frame: The tile it is cut to.

    Returns:
        observation: The product as its reader handed it back.
        sample: The crop the build would write.

    Raises:
        ValueError: When the product measured nothing or reaches none of the tile.
    """
    preprocessing = training_settings().preprocessing.get(name, {})
    observation = INSTRUMENTS[name].read_observation(identifier, **preprocessing)
    sample = None if observation is None else INSTRUMENTS[name].crop(observation, frame)
    if sample is None:
        raise ValueError(f"{name} holds nothing of tile {frame.name}.")
    return observation, sample


def acquisition_table(
    name: str, identifier: str, sample: Sample, frame: Tile
) -> widgets.HTML:
    """Tabulate what one built crop was taken under, as the build's index records it."""
    layout = INSTRUMENTS[name].layout
    path = str(sample_path(frame, name, identifier))
    record = observation_metadata(sample, frame, layout, identifier, path, None)
    if record.t_start is None:
        started = next(
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
    return panels.written("Acquisition", STATISTIC_VALUE, rows)


def model_input(name: str, sample: Sample) -> np.ndarray:
    """Return a built crop's values as the MAE loader hands them to the model.

    Args:
        name: The instrument, as the build names it.
        sample: The crop the build would write.

    Returns:
        values: Standardised by the training build's constants, CRISM clipped,
            mapped from -2 to 2 onto 0 to 1 to be drawn.
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
    standardised = (values - mean) / np.maximum(std, config.STD_FLOOR)
    if name == "CRISM":
        measured = sample.measured_ground[..., None] & np.isfinite(standardised)
        standardised = np.where(
            measured, np.clip(standardised, -config.CLIP, config.CLIP), 0.0
        )
    return np.clip((standardised + config.CLIP) / (2 * config.CLIP), 0.0, 1.0)


def min_max_input(
    identifier: str, sample: Sample, frame: Tile, per_band: bool = True
) -> np.ndarray:
    """Return a built CRISM crop's bands as the MAE loader scales them by range.

    Args:
        identifier: The observation the crop was cut from.
        sample: The crop the build would write.
        frame: The tile it is cut to.
        per_band: Whether each band takes its own range, or all share one.

    Returns:
        values: Each stored band mapped from the smallest to the largest value of
            this crop and of every other crop of the observation in the training
            index onto 0 to 1, zero where unmeasured or not finite.
    """
    layout = INSTRUMENTS["CRISM"].layout
    measured = measured_mask(
        sample.measured_ground, sample.measured_bands, layout, sample.cube
    )
    # This crop's own range, as the build writes it into its index row
    own = measured_statistics(sample.cube, measured, layout)
    others = [
        one
        for one in read_observation_metadata(
            built.dataset_root(training_settings().name)
        )
        if one.instrument == "CRISM"
        and one.identifier == identifier
        and one.tile != frame.name
    ]
    values = sample.cube.astype(layout.stored)
    # Mirrors read_band_ranges and normalised_values in mars-multisensor-mae's store
    lows = [own["band_min"], *(one.band_min for one in others)]
    highs = [own["band_max"], *(one.band_max for one in others)]
    low = np.fmin.reduce(np.array(lows, np.float64))
    high = np.fmax.reduce(np.array(highs, np.float64))
    if not per_band:
        low, high = np.nanmin(low), np.nanmax(high)
    scaled = (values - low.astype(np.float32)) / (high - low).astype(np.float32)
    kept = sample.measured_ground[..., None] & np.isfinite(scaled)
    return np.where(kept, scaled, 0.0)


def stretched(values: np.ndarray) -> np.ndarray:
    """Return values stretched from their 2nd to 98th percentile onto 0 to 1."""
    low, high = np.nanpercentile(values, config.STRETCH)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def side_by_side(
    images: Sequence[tuple[str, np.ndarray]],
    label: str | None = None,
    cmap: str = "gray",
) -> widgets.Image:
    """Draw images in one row, each under its own title.

    Args:
        images: Each image's title and values from 0 to 1, a colour axis last for RGB.
        label: What the row shows, written down its left side, or None for nothing.
        cmap: The colour map a single band is drawn in.

    Returns:
        row: The images as a PNG widget.
    """
    figure = Figure(figsize=(11, 4.5))
    if label is not None:
        figure.supylabel(label, fontsize=12)
    axes = figure.subplots(1, len(images), squeeze=False)[0]
    for axis, (text, values) in zip(axes, images, strict=True):
        axis.imshow(
            values, cmap=cmap, aspect="equal", interpolation="nearest", vmin=0, vmax=1
        )
        panels.titled(axis, text)
        axis.set_axis_off()
    figure.tight_layout()
    return panels.rendered(figure)
