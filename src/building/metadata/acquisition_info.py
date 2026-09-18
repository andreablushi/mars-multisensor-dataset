"""What one crop was taken under: where the Sun stood, where the spacecraft did."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from building.preprocessing.common.models.sample import Sample
from shared.maths import geodesy, physics
from shared.models.tile import Tile

LABEL_KEYS = {
    "incidence_deg": ("Incidence_angle",),
    "emission_deg": ("Emission_angle",),
    "phase_deg": ("Phase_angle",),
    "solar_longitude_deg": ("SOLAR_LONGITUDE", "SolarLongitude", "Solar_longitude"),
    "local_solar_time_h": ("Solar_time",),
    "solar_distance_km": ("SOLAR_DISTANCE", "Solar_distance"),
}

TARGET_DISTANCE_KEYS = ("TARGET_CENTER_DISTANCE",)


@dataclass(frozen=True, slots=True)
class AcquisitionInfo:
    """Where the Sun and the spacecraft stood over one crop, and in what season.

    Attributes:
        incidence_deg: The angle between the Sun and the areoid's normal, averaged
            over the samples the crop measured where its archive writes one per
            sample and the observation's own figure where it publishes one for the
            whole. A sounder's solar zenith angle is that same angle and is read
            as it.
        emission_deg: The angle between the spacecraft and that normal, read the
            same way.
        phase_deg: The angle the ground sees between the Sun and the spacecraft,
            read the same way.
        solar_longitude_deg: The areocentric longitude of the Sun, which is the
            season the observation was taken in.
        local_solar_time_h: The hour of the Martian day over the ground, on the
            twenty four hour clock.
        solar_distance_km: How far the Sun stood from Mars.
        spacecraft_altitude_km: How far the spacecraft stood above the ground,
            taken off the spheroid where an archive publishes its distance to the
            centre of Mars instead.
    """

    incidence_deg: float | None = None
    emission_deg: float | None = None
    phase_deg: float | None = None
    solar_longitude_deg: float | None = None
    local_solar_time_h: float | None = None
    solar_distance_km: float | None = None
    spacecraft_altitude_km: float | None = None


def acquisition_info(
    held: Sample, frame: Tile, measured: np.ndarray
) -> AcquisitionInfo:
    """Return what one crop was taken under, from its own planes and its label.

    Args:
        held: The crop, whose planes are reduced over the samples it measured and
            whose label is read for every quantity it carries no plane of.
        frame: The local frame of its tile, at whose latitude the spheroid an
            altitude stands above is measured.
        measured: Which of its samples are measurements inside that tile's box,
            over the ground axes alone, which its caller has already rooted.

    Returns:
        info: One scalar per quantity, each unset where neither a plane of the crop
            nor its label carries it.
    """
    collected = {}
    for one in fields(AcquisitionInfo):
        plane = getattr(held, one.name, None)
        collected[one.name] = (
            _measured_mean(plane, measured)
            if plane is not None
            else _label_scalar(held.label, LABEL_KEYS.get(one.name, ()))
        )
    if collected["spacecraft_altitude_km"] is None:
        centre = _label_scalar(held.label, TARGET_DISTANCE_KEYS)
        if centre is not None:
            stood = geodesy.spheroid_radius_m(frame.centre_lat) / physics.METRES_PER_KM
            collected["spacecraft_altitude_km"] = centre - stood
    return AcquisitionInfo(**collected)


def _measured_mean(plane: np.ndarray, measured: np.ndarray) -> float | None:
    """Return the mean of one plane over the samples its crop measured.

    Args:
        plane: The quantity at every sample of the crop, over its ground axes.
        measured: Which of those samples are measurements inside the tile's box.

    Returns:
        mean: The mean, or None where the crop measured no sample its archive
            wrote a finite value at.
    """
    kept = measured & np.isfinite(plane)
    return float(plane[kept].mean()) if kept.any() else None


def _label_scalar(label: dict[str, str], keys: tuple[str, ...]) -> float | None:
    """Return the first of the keys one label carries a number under.

    Args:
        label: The merged label of the observation.
        keys: What an archive may call the quantity, in the order they are
            preferred.

    Returns:
        value: The number, or None where the label carries none of them readably.
    """
    for key in keys:
        try:
            return float(label[key])
        except (KeyError, ValueError):
            continue
    return None
