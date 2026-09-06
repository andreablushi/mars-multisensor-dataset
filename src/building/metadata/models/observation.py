"""What one observation of one feature is, beside the arrays it was stored as."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    """One observation of one feature, and how to read the arrays beside it.

    Attributes:
        feature_class: The feature the observation was kept for.
        feature_name: The feature name as ODE spells it.
        instrument: The instrument that took it, as ODE names it.
        identifier: What that instrument was asked for, its observation or tile.
        path: Where its arrays were written, relative to the dataset's own root.
        axes: What each axis of the value array holds, in the array's own order.
        shape: The value array's shape, in that same order.
        ground_sample_m: How much ground one sample spans along each ground
            axis, in the order those axes run, measured off the position
            rather than claimed by a label.
        separable: Whether the position holds one axis each rather than a
            value per sample.
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
    t_start: datetime | None = None
    t_end: datetime | None = None
    altitude_min_m: float | None = None
    altitude_max_m: float | None = None
