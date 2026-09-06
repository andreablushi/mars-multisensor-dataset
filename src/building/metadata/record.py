"""Building what one stored observation is, from the position it was stored with."""

from __future__ import annotations

from datetime import datetime

from building.common.layout import Layout
from building.metadata.models.feature import FeatureFrame
from building.metadata.models.observation import ObservationRecord
from building.preprocessing.common import relative_positioning
from building.preprocessing.common.models.sample import Sample


def observation_record(
    held: Sample,
    frame: FeatureFrame,
    layout: Layout,
    path: str,
    t_start: datetime | None = None,
    t_end: datetime | None = None,
    altitude: tuple[float, float] | None = None,
) -> ObservationRecord:
    """Return the record one stored observation is read back through.

    Args:
        held: The sample that was written, whose position the ground sample is
            measured off.
        frame: The local frame of the feature it was kept for.
        layout: What its instrument's arrays hold.
        path: Where its arrays were written, relative to the dataset's own root.
        t_start: When it started, or None where the archive publishes no time.
        t_end: When it ended, or None for the same reason.
        altitude: How low and how high the spacecraft was, for a sounder whose
            delay axis is read through it, and None for every other instrument.

    Returns:
        The record, its ground sample measured rather than claimed.
    """
    low, high = altitude if altitude else (None, None)
    return ObservationRecord(
        feature_class=frame.feature_class,
        feature_name=frame.feature_name,
        instrument=layout.instrument,
        identifier=held.identifier,
        path=path,
        axes=layout.axes,
        shape=tuple(getattr(held, layout.measurement).shape),
        ground_sample_m=relative_positioning.ground_sample_m(held.position, frame),
        separable=held.position.separable,
        t_start=t_start,
        t_end=t_end,
        altitude_min_m=low,
        altitude_max_m=high,
    )
