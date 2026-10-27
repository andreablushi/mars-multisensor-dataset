"""What one crop was taken under: where the Sun stood, where the spacecraft did."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime

import numpy as np

from building.preprocessing.common.models.sample import Sample
from common.maths import geodesy, physics, sun
from common.models.tile import Tile

LABEL_KEYS = {
    "incidence_deg": ("Incidence_angle",),
    "emission_deg": ("Emission_angle",),
    "phase_deg": ("Phase_angle",),
    "solar_longitude_deg": ("SOLAR_LONGITUDE", "SolarLongitude", "Solar_longitude"),
    "local_solar_time_h": ("Solar_time",),
    "solar_distance_km": ("SOLAR_DISTANCE", "Solar_distance"),
}

TARGET_DISTANCE_KEYS = ("TARGET_CENTER_DISTANCE",)

IMAGE_AZIMUTH_KEYS = ("NORTH_AZIMUTH", "SUB_SOLAR_AZIMUTH")

COMPUTED = {"CRISM": ("solar_azimuth_deg",), "SHARAD": ("solar_longitude_deg",)}


@dataclass(frozen=True, slots=True)
class AcquisitionInfo:
    """Where the Sun and the spacecraft stood over one crop, and in what season.

    Attributes:
        incidence_deg: The angle between the Sun and the areoid's normal.
        emission_deg: The angle between the spacecraft and that normal.
        phase_deg: The angle between the Sun and the spacecraft seen from ground.
        solar_longitude_deg: The areocentric longitude of the Sun.
        local_solar_time_h: The local Martian hour over the ground, of 24.
        solar_distance_km: How far the Sun stood from Mars.
        spacecraft_altitude_km: How high the spacecraft stood above the ground.
        solar_azimuth_deg: Where the Sun stood seen from the crop, clockwise from
            north.
    """

    incidence_deg: float | None = None
    emission_deg: float | None = None
    phase_deg: float | None = None
    solar_longitude_deg: float | None = None
    local_solar_time_h: float | None = None
    solar_distance_km: float | None = None
    spacecraft_altitude_km: float | None = None
    solar_azimuth_deg: float | None = None


def acquisition_info(
    held: Sample,
    frame: Tile,
    instrument: str,
    t_start: datetime | None,
    t_end: datetime | None,
) -> AcquisitionInfo:
    """Return what one crop was taken under, from its planes, its label and its time.

    Args:
        held: The crop, reduced over its measured samples.
        frame: The local frame of its tile.
        instrument: The instrument that took it, as ODE names it.
        t_start: When it started, or None.
        t_end: When it ended, or None.

    Returns:
        info: One scalar per quantity, unset where the crop carries none.
    """
    measured = held.measured_ground
    collected = {}
    for one in fields(AcquisitionInfo):
        plane = getattr(held, one.name, None)
        if plane is None:
            collected[one.name] = _label_scalar(
                held.label, LABEL_KEYS.get(one.name, ())
            )
            continue
        kept = measured & np.isfinite(plane)
        collected[one.name] = float(plane[kept].mean()) if kept.any() else None
    distance = _label_scalar(held.label, TARGET_DISTANCE_KEYS)
    if collected["spacecraft_altitude_km"] is None and distance is not None:
        radius = geodesy.spheroid_radius_m(frame.centre_lat) / physics.METRES_PER_KM
        collected["spacecraft_altitude_km"] = distance - radius
    north, toward_sun = (
        _label_scalar(held.label, (key,)) for key in IMAGE_AZIMUTH_KEYS
    )
    if north is not None and toward_sun is not None:
        collected["solar_azimuth_deg"] = ground_solar_azimuth_deg(north, toward_sun)
    collected.update(computed_acquisition(instrument, frame, t_start, t_end))
    return AcquisitionInfo(**collected)


def ground_solar_azimuth_deg(north_azimuth: float, sub_solar_azimuth: float) -> float:
    """Return the Sun's bearing from north, from two clockwise image plane azimuths."""
    return (sub_solar_azimuth - north_azimuth) % geodesy.TURN


def computed_acquisition(
    instrument: str, frame: Tile, t_start: datetime | None, t_end: datetime | None
) -> dict[str, float]:
    """Return the quantities `COMPUTED` names for one instrument, at its tile's centre.

    Args:
        instrument: The instrument, as ODE names it.
        frame: The local frame of the crop's tile.
        t_start: When the crop started, or None.
        t_end: When it ended, or None.

    Returns:
        computed: One value per quantity `COMPUTED` names, none without a time.
    """
    if t_start is None:
        return {}
    moment = t_start if t_end is None else t_start + (t_end - t_start) / 2
    values = {
        "solar_longitude_deg": sun.solar_longitude_deg(moment),
        "solar_azimuth_deg": sun.solar_azimuth_deg(
            moment, frame.centre_lat, frame.centre_lon
        ),
    }
    return {name: values[name] for name in COMPUTED.get(instrument, ())}


def _label_scalar(label: dict[str, str], keys: tuple[str, ...]) -> float | None:
    """Return the first of the keys one label carries a number under.

    Args:
        label: The merged label of the observation.
        keys: What an archive may call the quantity, preferred first.

    Returns:
        value: The number, or None where the label carries none of them readably.
    """
    for key in keys:
        try:
            return float(label[key])
        except (KeyError, ValueError):
            continue
    return None
