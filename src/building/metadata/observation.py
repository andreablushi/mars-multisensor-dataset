"""What the dataset says about one stored observation, beside the arrays it holds."""

from __future__ import annotations

import warnings
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
        valid_count: How many stored values are measurements, at least one.
        value_min: The smallest of those values.
        value_max: The largest of them.
        value_mean: Their mean.
        value_std: Their standard deviation.
        band_mean: The mean of each spectral band, None for a band it never
            measured, or None without wavelength.
        band_std: Each band's standard deviation, None likewise.
        band_valid_count: The measurements each band pools, or None.
        t_start: When the observation started, or None.
        t_end: When it ended, or None for the same reason.
        acquisition: Where the Sun and the spacecraft stood over the crop.
        normalized: Whether its values are standardised by the dataset's constants.
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
    value_min: float
    value_max: float
    value_mean: float
    value_std: float
    band_mean: tuple[float | None, ...] | None = None
    band_std: tuple[float | None, ...] | None = None
    band_valid_count: tuple[int, ...] | None = None
    t_start: datetime | None = None
    t_end: datetime | None = None
    acquisition: AcquisitionInfo = field(default_factory=AcquisitionInfo)
    normalized: bool = False

    @property
    def identity(self) -> tuple[str, str, str]:
        """Return the tile it was kept for, and the product it was cut from."""
        return (self.tile, self.instrument, self.identifier)


def observation_metadata(
    held: Sample,
    frame: Tile,
    layout: Layout,
    identifier: str,
    path: str,
    t_start: datetime | None,
) -> ObservationMetadata:
    """Return what one stored observation is read back through.

    Args:
        held: The sample that was written, which measured something.
        frame: The local frame of the tile it was kept for.
        layout: What its instrument's arrays hold.
        identifier: What that instrument was asked for, its observation or sheet.
        path: Where its arrays were written, relative to the dataset's own root.
        t_start: When it started, for an archive whose label publishes no time.

    Returns:
        metadata: The metadata, measured rather than claimed.
    """
    values = getattr(held, layout.measurement)
    measured = measured_mask(held.measured_ground, held.measured_bands, layout, values)
    return ObservationMetadata(
        tile=frame.name,
        instrument=layout.instrument,
        identifier=identifier,
        path=path,
        axes=layout.axes,
        shape=tuple(values.shape),
        sample_spacing_m=relative_positioning.sample_spacing_m(held.position, frame),
        separable=held.position.separable,
        t_start=_label_time(held.label, STARTED) or t_start,
        t_end=_label_time(held.label, STOPPED),
        acquisition=acquisition_info(held, frame),
        **measured_statistics(values, measured, layout),
    )


def measured_mask(
    measured_ground: np.ndarray,
    measured_bands: np.ndarray | None,
    layout: Layout,
    values: np.ndarray,
) -> np.ndarray:
    """Return where the value array holds a measurement, on its own shape.

    Args:
        measured_ground: Whether each ground sample measured, over the ground axes.
        measured_bands: Whether each band measured, or None without wavelength.
        layout: What the instrument's arrays hold.
        values: The value array.

    Returns:
        measured: The mask, broadcast to the value array's shape.
    """
    # A ground mask reaches every value on it, so it spreads over the instrument's axes.
    measured = _spread_mask(measured_ground, layout.axis_indices(Axis.GROUND), values)
    # A band the observation never measured holds nothing, whatever the ground says.
    if measured_bands is not None:
        wavelength = layout.axis_indices(Axis.WAVELENGTH)
        measured = measured & _spread_mask(measured_bands, wavelength, values)
    return measured


def measured_statistics(
    values: np.ndarray, measured: np.ndarray, layout: Layout
) -> dict[str, object]:
    """Return the index fields that describe the measured values.

    Args:
        values: The value array.
        measured: Where it holds a measurement, on its own shape, somewhere True.
        layout: What the instrument's arrays hold.

    Returns:
        fields: The count, min, max, mean and std, and each band's where it has one.
    """
    ground = layout.axis_indices(Axis.GROUND)
    # An integer holds no infinite identity, so the reduction starts at its type's edge.
    limits = (
        np.iinfo(values.dtype)
        if np.issubdtype(values.dtype, np.integer)
        else np.finfo(values.dtype)
    )
    band_mean = band_std = band_valid_count = None
    # A band is the one axis a reader normalises against, so it survives the reduction.
    if Axis.WAVELENGTH in layout.axes:
        band_valid_count = tuple(measured.sum(axis=ground).tolist())
        # An unmeasured band is an empty slice, whose statistics are None
        with warnings.catch_warnings(action="ignore"):
            reduced = [
                np.mean(values, axis=ground, where=measured),
                np.std(values, axis=ground, where=measured),
            ]
        band_mean, band_std = (
            tuple(
                one if pooled else None
                for one, pooled in zip(each.tolist(), band_valid_count, strict=True)
            )
            for each in reduced
        )
    return {
        "valid_count": int(measured.sum()),
        "value_min": float(np.min(values, where=measured, initial=limits.max)),
        "value_max": float(np.max(values, where=measured, initial=limits.min)),
        "value_mean": float(np.mean(values, where=measured)),
        "value_std": float(np.std(values, where=measured)),
        "band_mean": band_mean,
        "band_std": band_std,
        "band_valid_count": band_valid_count,
    }


def _spread_mask(
    mask: np.ndarray, axes: tuple[int, ...], values: np.ndarray
) -> np.ndarray:
    """Return one mask over some axes of the value array, spread over every value.

    Args:
        mask: The flags over those axes alone, in the array's own order.
        axes: Where the axes the mask runs along sit in the value array.
        values: The value array it is spread over.

    Returns:
        spread: The mask, broadcast to the value array's shape.
    """
    shape = tuple(size if at in axes else 1 for at, size in enumerate(values.shape))
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
