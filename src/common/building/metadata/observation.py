"""What the dataset says about one stored observation, beside the arrays it holds."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from common.building.common.layout import GROUND, WAVELENGTH, Layout
from common.building.common.pds import times
from common.building.metadata.acquisition_info import AcquisitionInfo, acquisition_info
from common.building.preprocessing.common import relative_positioning
from common.building.preprocessing.common.models.sample import Sample
from common.disk import parquet
from common.models.tile import Tile

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
        sample_spacing_m: How much ground one sample spans along each ground axis,
            in the order those axes run, measured rather than claimed.
        separable: Whether the grid it was placed on holds one ground axis each
            rather than a pair per sample, which the stored offsets no longer
            do either way.
        valid_count: How many of the stored values are measurements, which is what
            the statistics beside it were measured over and pool by.
        value_min: The smallest of those values, or None where the crop holds
            no measurement to say anything about.
        value_max: The largest of them, or None for the same reason.
        value_mean: Their mean, or None for the same reason.
        value_std: Their standard deviation, or None for the same reason.
        band_mean: The mean of each spectral band on its own, reduced over the
            ground axes alone, for an instrument whose axes hold a wavelength,
            and None for every other and where the crop measures nothing.
        band_std: Each band's standard deviation, or None for the same reasons.
        band_valid_count: How many measurements each of those bands pools, or None
            for the same reasons, and zero for a band the observation never measured.
        t_start: When the observation started, or None where the archive
            publishes no time for it.
        t_end: When it ended, or None for the same reason.
        acquisition: Where the Sun and the spacecraft stood over the crop, every
            quantity of it unset for an archive that publishes none.
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
        """Return what tells this stored observation from every other.

        Returns:
            identity: The tile it was kept for, and the product it was cut from.
        """
        return (self.tile, self.instrument, self.identifier)


def observation_metadata(
    held: Sample,
    frame: Tile,
    layout: Layout,
    path: str,
    t_start: datetime | None = None,
) -> ObservationMetadata:
    """Return what one stored observation is read back through.

    Args:
        held: The sample that was written, whose position the ground sample is
            measured off and whose label the times are read from.
        frame: The local frame of the tile it was kept for.
        layout: What its instrument's arrays hold.
        path: Where its arrays were written, relative to the dataset's own root.
        t_start: When it started, for an archive whose label publishes no time.

    Returns:
        metadata: The metadata, its ground sample and statistics measured rather than
            claimed, and those unset where it holds no measurement.
    """
    values = getattr(held, layout.measurement)
    # A ground mask reaches every value on it, so it spreads over the instrument's axes.
    ground = tuple(
        size if holds == GROUND else 1
        for size, holds in zip(values.shape, layout.axes, strict=True)
    )
    measured_ground = held.measured_ground
    on_ground = np.broadcast_to(measured_ground.reshape(ground), values.shape)
    measured = on_ground
    # A band the observation never measured holds nothing, whatever the ground says.
    if held.measured_bands is not None:
        bands = tuple(
            size if holds == WAVELENGTH else 1
            for size, holds in zip(values.shape, layout.axes, strict=True)
        )
        measured = on_ground & held.measured_bands.reshape(bands)
    # An integer holds no infinite identity, so the reduction starts at its type's edge.
    limits = (
        np.iinfo(values.dtype)
        if np.issubdtype(values.dtype, np.integer)
        else np.finfo(values.dtype)
    )
    counted = int(measured.sum())
    # A crop can reach the box and measure nothing, and nothing says nothing
    smallest, largest, mean, deviation = (
        (
            float(np.min(values, where=measured, initial=limits.max)),
            float(np.max(values, where=measured, initial=limits.min)),
            float(np.mean(values, where=measured)),
            float(np.std(values, where=measured)),
        )
        if counted
        else (None, None, None, None)
    )
    # A band is the one axis a reader normalises against, so it survives the reduction.
    over = tuple(axis for axis, holds in enumerate(layout.axes) if holds == GROUND)
    band_mean, band_std, band_valid_count = None, None, None
    if counted and WAVELENGTH in layout.axes:
        pooled = measured.sum(axis=over)
        band_mean = tuple(np.mean(values, axis=over, where=on_ground).tolist())
        band_std = tuple(np.std(values, axis=over, where=on_ground).tolist())
        band_valid_count = tuple(pooled.tolist())
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
        value_min=smallest,
        value_max=largest,
        value_mean=mean,
        value_std=deviation,
        band_mean=band_mean,
        band_std=band_std,
        band_valid_count=band_valid_count,
        t_start=_moment(held.label, STARTED) or t_start,
        t_end=_moment(held.label, STOPPED),
        acquisition=acquisition_info(held, frame, measured_ground),
    )


def _moment(label: dict[str, str], key: str) -> datetime | None:
    """Return one time the label names, or None where it names none it can read.

    Args:
        label: The merged label of the observation.
        key: Which time to read.

    Returns:
        moment: The time in UTC, or None where the label holds no readable one.
    """
    held = label.get(key)
    try:
        return times.moment(held) if held else None
    except ValueError:
        return None


SCHEMA = parquet.schema_of(ObservationMetadata)
