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
from building.configs import sharad
from building.metadata.dataset import read_normalization
from building.metadata.index import read_observation_metadata
from building.metadata.observation import (
    measured_mask,
    measured_statistics,
    observation_metadata,
)
from building.models.instrument import INSTRUMENTS, observation_identifier
from building.preprocessing.common.models.position import Position
from building.preprocessing.common.models.sample import Sample
from building.preprocessing.common.store import sample_path
from common.config import training_settings
from common.models.tile import Tile
from common.pds import images
from visualization import panels
from visualization.config import STATISTIC_VALUE
from visualization.training_qualitative import config

Layer = tuple[Position, np.ndarray]


def observed_sample(
    name: str, identifier: str, frame: Tile, **settings: object
) -> tuple[Any, Sample]:
    """Read one fetched product and cut it to a tile, as a training build does.

    Args:
        name: The instrument, as the build names it.
        identifier: What that instrument was asked for, its observation.
        frame: The tile it is cut to.
        **settings: Preprocessing settings standing in for the build's own.

    Returns:
        observation: The product as its reader handed it back.
        sample: The crop the build would write.

    Raises:
        ValueError: When the product measured nothing or reaches none of the tile.
    """
    preprocessing = {**training_settings().preprocessing.get(name, {}), **settings}
    observation = INSTRUMENTS[name].read_observation(identifier, **preprocessing)
    sample = None if observation is None else INSTRUMENTS[name].crop(observation, frame)
    if sample is None:
        raise ValueError(f"{name} holds nothing of tile {frame.name}.")
    return observation, sample


def indexed_identifiers(name: str, frame: Tile) -> tuple[str, ...]:
    """Return what the training index holds of one instrument on a tile, by start."""
    held = [
        one
        for one in read_observation_metadata(
            built.dataset_root(training_settings().name)
        )
        if one.tile == frame.name and one.instrument == name
    ]
    return tuple(one.identifier for one in sorted(held, key=lambda one: one.t_start))


def acquisition_values(
    name: str, identifier: str, sample: Sample, frame: Tile
) -> list[tuple[str, str]]:
    """Return what one built crop was taken under, as the build's index records it.

    Args:
        name: The instrument, as the build names it.
        identifier: The product the crop was cut from.
        sample: The crop the build would write.
        frame: The tile it is cut to.

    Returns:
        rows: Each field beside its value, written out.
    """
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
    return [
        ("Tile", frame.name),
        ("Started", str(record.t_start)),
        ("Ended", str(record.t_end)),
        *(
            (field, f"{value:.4g}")
            for field, value in asdict(record.acquisition).items()
            if value is not None
        ),
    ]


def acquisition_table(
    name: str, identifiers: Sequence[str], samples: Sequence[Sample], frame: Tile
) -> widgets.HTML:
    """Tabulate what every built crop was taken under, one column per product."""
    columns = [
        dict(acquisition_values(name, identifier, sample, frame))
        for identifier, sample in zip(identifiers, samples, strict=True)
    ]
    fields = dict.fromkeys(field for column in columns for field in column)
    rows = [(field, *(column.get(field, "") for column in columns)) for field in fields]
    return panels.written("Acquisition", (STATISTIC_VALUE[0], *identifiers), rows)


def standardised_values(
    values: np.ndarray, axes: Sequence[str], name: str
) -> np.ndarray:
    """Return values standardised by the training build's constants.

    Args:
        values: The values, as the build stored them.
        axes: What each axis of the values holds.
        name: The instrument, or `clutter` for the SHARAD cluttergram.

    Returns:
        values: Less the mean and over the std, per band where they have bands.
    """
    constants = read_normalization(built.dataset_root(training_settings().name))[name]
    shape = [-1 if axis == Axis.WAVELENGTH else 1 for axis in axes]
    mean, std = (
        np.asarray(constants[key], np.float32).reshape(shape) for key in ("mean", "std")
    )
    return (values - mean) / np.maximum(std, config.STD_FLOOR)


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
    # Mirrors standardised_values and read_observation in mars-multisensor-mae's store
    standardised = standardised_values(
        values.astype(layout.stored or values.dtype), layout.axes, name
    )
    if name == "CRISM":
        measured = sample.measured_ground[..., None] & np.isfinite(standardised)
        standardised = np.where(
            measured, np.clip(standardised, -config.CLIP, config.CLIP), 0.0
        )
    return np.clip((standardised + config.CLIP) / (2 * config.CLIP), 0.0, 1.0)


def sharad_model_input(preprocessed: np.ndarray) -> np.ndarray:
    """Return a preprocessed SHARAD array standardised by the radargram's constants."""
    return standardised_values(preprocessed, sharad.LAYOUT.axes, "SHARAD")


def sharad_power(identifier: str) -> np.ndarray:
    """Return a whole SHARAD radargram as archived, delay rows by traces."""
    return images.load_cube(
        sharad.CACHE.product_files(identifier, sharad.Kind.OBSERVATION)[".img"]
    )[0][:, :, 0]


def sharad_gate_name(**settings: float) -> str:
    """Return the build's SHARAD gate settings, any given in their place, as one name.

    Args:
        **settings: Gate settings standing in for the build's own.

    Returns:
        name: Every setting beside its value, to cache a panel under.
    """
    gate = {**training_settings().preprocessing["SHARAD"], **settings}
    return "_".join(f"{key}_{value:g}" for key, value in gate.items())


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


def stretched_alike(layers: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Return arrays stretched from their pooled 2nd to 98th percentile onto 0 to 1."""
    pooled = np.concatenate([layer.ravel() for layer in layers])
    low, high = np.nanpercentile(pooled, config.STRETCH)
    return [np.clip((layer - low) / (high - low), 0.0, 1.0) for layer in layers]


def stretched(values: np.ndarray) -> np.ndarray:
    """Return values stretched from their 2nd to 98th percentile onto 0 to 1."""
    return stretched_alike([values])[0]


def side_by_side(
    mosaics: Sequence[tuple[str, Sequence[Layer]]],
    label: str | None = None,
    cmap: str = "gray",
    height: float = config.ROW_HEIGHT,
) -> widgets.Image:
    """Draw mosaics in one row, each under its own title, every crop over the last.

    Args:
        mosaics: Each mosaic's title and crops, oldest first, each a position beside
            values from 0 to 1, a colour axis last for RGB, NaN where unmeasured.
        label: What the row shows, written down its left side, or None for nothing.
        cmap: The colour map a single band is drawn in.
        height: How tall the row is drawn, in inches.

    Returns:
        row: The mosaics as a PNG widget.
    """
    figure = Figure(figsize=(11, height))
    if label is not None:
        figure.supylabel(label, fontsize=12)
    axes = figure.subplots(1, len(mosaics), squeeze=False)[0]
    drawn = [
        [thinned(position, values) for position, values in crops]
        for _, crops in mosaics
    ]
    every = [ground for crops in drawn for ground in crops]
    (west, east), (south, north) = (
        (
            min(np.nanmin(ground[at]) for ground in every),
            max(np.nanmax(ground[at]) for ground in every),
        )
        for at in (0, 1)
    )
    grid = next(position.grid for _, crops in mosaics for position, _ in crops)
    aspect = 1.0 if grid else 1.0 / np.cos(np.radians((south + north) / 2.0))
    for axis, (text, _), crops in zip(axes, mosaics, drawn, strict=True):
        for eastings, northings, values in crops:
            if values.ndim == 3:
                shown = np.dstack(
                    [
                        np.clip(np.nan_to_num(values), 0, 1),
                        np.isfinite(values).any(axis=-1),
                    ]
                )
                axis.pcolormesh(eastings, northings, shown, shading="nearest")
            else:
                axis.pcolormesh(
                    eastings,
                    northings,
                    np.ma.masked_invalid(values),
                    shading="nearest",
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                )
        axis.set_xlim(west, east)
        axis.set_ylim(south, north)
        axis.set_aspect(aspect)
        panels.titled(axis, text)
        axis.set_axis_off()
    figure.tight_layout()
    return panels.rendered(figure)


def thinned(
    position: Position, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one crop's eastings, northings and values, thinned to be drawn.

    Args:
        position: Where the crop's samples sit.
        values: The crop's values over its ground axes, a colour axis last for RGB.

    Returns:
        eastings: The easting of every sample kept, lines by samples.
        northings: Its northing, on the same grid.
        values: Its values, thinned alike.
    """
    stride = -(-max(position.sizes) // config.DRAWN_PIXELS)
    taken = (slice(None, None, stride),) * 2
    northings, eastings = np.broadcast_arrays(*position.crossed_part(taken))
    return eastings, northings, values[taken]
