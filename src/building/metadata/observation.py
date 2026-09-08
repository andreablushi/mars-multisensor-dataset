"""What the dataset says about one stored observation, beside the arrays it holds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from building.common.layout import GROUND, Layout
from building.common.pds import times
from building.models.feature import FeatureFrame
from building.preprocessing.common import relative_positioning
from building.preprocessing.common.models.sample import Sample
from shared.disk import parquet

# What a label calls the two ends of the time a product was taken over.
STARTED = "START_TIME"
STOPPED = "STOP_TIME"


@dataclass(frozen=True, slots=True)
class ObservationMetadata:
    """One observation of one feature, and how to read the arrays beside it.

    Attributes:
        feature_class: The feature the observation was kept for.
        feature_name: The feature name as ODE spells it.
        instrument: The instrument that took it, as ODE names it.
        identifier: What that instrument was asked for, its observation or tile.
        path: Where its arrays were written, relative to the dataset's own root.
        axes: What each axis of the value array holds, in the array's own order.
        shape: The value array's shape, in that same order.
        ground_sample_m: How much ground one sample spans along each ground axis,
            in the order those axes run, measured rather than claimed.
        separable: Whether the position holds one axis each rather than a
            value per sample.
        valid_count: How many of the stored values are measurements, which is what
            the statistics beside it were measured over and pool by.
        value_min: The smallest of those values, or None where the crop holds
            no measurement to say anything about.
        value_max: The largest of them, or None for the same reason.
        value_mean: Their mean, or None for the same reason.
        value_std: Their standard deviation, or None for the same reason.
        t_start: When the observation started, or None where the archive
            publishes no time for it.
        t_end: When it ended, or None for the same reason.
        altitude_min_m: How low the spacecraft was above the ground, for a
            sounder whose delay axis is read through it, and None otherwise.
        altitude_max_m: How high it was, for the same instrument.
    """

    feature_class: str
    feature_name: str
    instrument: str
    identifier: str
    path: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    ground_sample_m: tuple[float, ...]
    separable: bool
    valid_count: int
    value_min: float | None
    value_max: float | None
    value_mean: float | None
    value_std: float | None
    t_start: datetime | None = None
    t_end: datetime | None = None
    altitude_min_m: float | None = None
    altitude_max_m: float | None = None

    @property
    def feature(self) -> tuple[str, str]:
        """Return the feature this observation was kept for.

        Returns:
            feature: Its class and its name.
        """
        return (self.feature_class, self.feature_name)

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """Return what tells this stored observation from every other.

        Returns:
            identity: The feature it was kept for, and the product it was cut from.
        """
        return (*self.feature, self.instrument, self.identifier)


def observation_metadata(
    held: Sample,
    frame: FeatureFrame,
    layout: Layout,
    path: str,
    t_start: datetime | None = None,
    altitude: tuple[float, float] | None = None,
) -> ObservationMetadata:
    """Return what one stored observation is read back through.

    Args:
        held: The sample that was written, whose position the ground sample is
            measured off and whose label the times are read from.
        frame: The local frame of the feature it was kept for.
        layout: What its instrument's arrays hold.
        path: Where its arrays were written, relative to the dataset's own root.
        t_start: When it started, for an archive whose label publishes no time.
        altitude: How low and how high the spacecraft was, for a sounder whose
            delay axis is read through it, and None for every other instrument.

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
    measured = np.ones(ground, dtype=bool)
    for mask in (held.inside, held.valid):
        if mask is not None:
            measured = measured & mask.reshape(ground)
    low, high = altitude if altitude else (None, None)
    # An integer holds no infinite identity, so the reduction starts at its type's edge.
    limits = (
        np.iinfo(values.dtype)
        if np.issubdtype(values.dtype, np.integer)
        else np.finfo(values.dtype)
    )
    counted = int(measured.sum()) * int(np.prod(values.shape) // measured.size)
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
    return ObservationMetadata(
        feature_class=frame.feature_class,
        feature_name=frame.feature_name,
        instrument=layout.instrument,
        identifier=held.identifier,
        path=path,
        axes=layout.axes,
        shape=tuple(values.shape),
        ground_sample_m=relative_positioning.ground_sample_m(held.position, frame),
        separable=held.position.separable,
        valid_count=counted,
        value_min=smallest,
        value_max=largest,
        value_mean=mean,
        value_std=deviation,
        t_start=_moment(held.label, STARTED) or t_start,
        t_end=_moment(held.label, STOPPED),
        altitude_min_m=low,
        altitude_max_m=high,
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
