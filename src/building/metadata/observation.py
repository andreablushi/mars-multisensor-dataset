"""What the dataset says about one stored observation, beside the arrays it holds."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from building.common.layout import Axis, Layout
from building.metadata.acquisition_info import AcquisitionInfo, acquisition_info
from building.preprocessing.common import relative_positioning
from building.preprocessing.common.models.sample import Sample
from common.disk import parquet
from common.models.tile import Tile
from common.pds.tables import parse_timestamp

# What a label calls the two ends of the time a product was taken over.
STARTED = "START_TIME"
STOPPED = "STOP_TIME"


@dataclass(frozen=True, slots=True)
class ObservationMetadata:
    """One observation of one tile, and how to read the arrays beside it.

    Attributes:
        tile: The name of the tile the observation was kept for.
        instrument: The instrument that took it, as ODE names it.
        identifier: What that instrument was asked for, its observation or sheet.
        path: Where its arrays were written, relative to the dataset's own root.
        axes: What each axis of the value array holds, in the array's own order.
        shape: The value array's shape, in that same order.
        sample_spacing_m: The measured ground one sample spans per ground axis.
        separable: Whether its grid held one ground axis each.
        valid_count: How many stored values are measurements.
        value_min: The smallest of those values, or None.
        value_max: The largest of them, or None for the same reason.
        value_mean: Their mean, or None for the same reason.
        value_std: Their standard deviation, or None for the same reason.
        band_mean: The mean of each spectral band, or None without wavelength.
        band_std: Each band's standard deviation, or None for the same reasons.
        band_valid_count: The measurements each band pools, or None.
        t_start: When the observation started, or None.
        t_end: When it ended, or None for the same reason.
        acquisition: Where the Sun and the spacecraft stood over the crop.
    """

    tile: str
    instrument: str
    identifier: str
    path: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    sample_spacing_m: tuple[float, ...]
    separable: bool
    valid_count: int
    value_min: float | None
    value_max: float | None
    value_mean: float | None
    value_std: float | None
    band_mean: tuple[float, ...] | None = None
    band_std: tuple[float, ...] | None = None
    band_valid_count: tuple[int, ...] | None = None
    t_start: datetime | None = None
    t_end: datetime | None = None
    acquisition: AcquisitionInfo = field(default_factory=AcquisitionInfo)

    @property
    def identity(self) -> tuple[str, str, str]:
        """Return the tile it was kept for, and the product it was cut from."""
        return (self.tile, self.instrument, self.identifier)


def observation_metadata(
    held: Sample,
    frame: Tile,
    layout: Layout,
    path: str,
    t_start: datetime | None,
) -> ObservationMetadata:
    """Return what one stored observation is read back through.

    Args:
        held: The sample that was written.
        frame: The local frame of the tile it was kept for.
        layout: What its instrument's arrays hold.
        path: Where its arrays were written, relative to the dataset's own root.
        t_start: When it started, for an archive whose label publishes no time.

    Returns:
        metadata: The metadata, measured rather than claimed.
    """
    values = getattr(held, layout.measurement)
    # A ground mask reaches every value on it, so it spreads over the instrument's axes.
    on_ground = _spread_mask(held.measured_ground, Axis.GROUND, values, layout)
    measured = on_ground
    # A band the observation never measured holds nothing, whatever the ground says.
    if held.measured_bands is not None:
        bands = _spread_mask(held.measured_bands, Axis.WAVELENGTH, values, layout)
        measured = on_ground & bands
    # An integer holds no infinite identity, so the reduction starts at its type's edge.
    limits = (
        np.iinfo(values.dtype)
        if np.issubdtype(values.dtype, np.integer)
        else np.finfo(values.dtype)
    )
    counted = int(measured.sum())
    value_min = value_max = value_mean = value_std = None
    # A crop can reach the box and measure nothing, and nothing says nothing
    if counted:
        value_min = float(np.min(values, where=measured, initial=limits.max))
        value_max = float(np.max(values, where=measured, initial=limits.min))
        value_mean = float(np.mean(values, where=measured))
        value_std = float(np.std(values, where=measured))
    band_mean = band_std = band_valid_count = None
    # A band is the one axis a reader normalises against, so it survives the reduction.
    if counted and Axis.WAVELENGTH in layout.axes:
        over = tuple(
            axis for axis, holds in enumerate(layout.axes) if holds == Axis.GROUND
        )
        band_mean = tuple(np.mean(values, axis=over, where=on_ground).tolist())
        band_std = tuple(np.std(values, axis=over, where=on_ground).tolist())
        band_valid_count = tuple(measured.sum(axis=over).tolist())
    return ObservationMetadata(
        tile=frame.name,
        instrument=layout.instrument,
        identifier=held.identifier,
        path=path,
        axes=layout.axes,
        shape=tuple(values.shape),
        sample_spacing_m=relative_positioning.sample_spacing_m(held.position, frame),
        separable=held.position.separable,
        valid_count=counted,
        value_min=value_min,
        value_max=value_max,
        value_mean=value_mean,
        value_std=value_std,
        band_mean=band_mean,
        band_std=band_std,
        band_valid_count=band_valid_count,
        t_start=_label_time(held.label, STARTED) or t_start,
        t_end=_label_time(held.label, STOPPED),
        acquisition=acquisition_info(held, frame),
    )


def _spread_mask(
    mask: np.ndarray, kind: Axis, values: np.ndarray, layout: Layout
) -> np.ndarray:
    """Return one mask over the axes of one kind, spread over every value.

    Args:
        mask: The flags over those axes alone, in the array's own order.
        kind: What the axes the mask runs along hold.
        values: The value array it is spread over.
        layout: What each axis of that array holds.

    Returns:
        spread: The mask, broadcast to the value array's shape.
    """
    shape = tuple(
        size if holds == kind else 1
        for size, holds in zip(values.shape, layout.axes, strict=True)
    )
    return np.broadcast_to(mask.reshape(shape), values.shape)


def _label_time(label: dict[str, str], key: str) -> datetime | None:
    """Return one time the label names, or None where it names none it can read.

    Args:
        label: The merged label of the observation.
        key: Which time to read.

    Returns:
        moment: The time in UTC, or None where the label holds no readable one.
    """
    text = label.get(key)
    try:
        return parse_timestamp(text) if text else None
    except ValueError:
        return None


SCHEMA = parquet.schema_of(ObservationMetadata)
