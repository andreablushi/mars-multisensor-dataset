"""What one crop was taken under: where the Sun stood, where the spacecraft did."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from building.preprocessing.common.models.sample import Sample
from common.maths import geodesy, physics
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
        held: The crop, reduced over its measured samples.
        frame: The local frame of its tile.
        measured: Which samples are measurements inside the tile's box.

    Returns:
        info: One scalar per quantity, unset where the crop carries none.
    """
    collected = {}
    for one in fields(AcquisitionInfo):
        plane = getattr(held, one.name, None)
        collected[one.name] = (
            _measured_mean(plane, measured)
            if plane is not None
            else _label_scalar(held.label, LABEL_KEYS.get(one.name, ()))
        )
    distance = _label_scalar(held.label, TARGET_DISTANCE_KEYS)
    if collected["spacecraft_altitude_km"] is None and distance is not None:
        radius = geodesy.spheroid_radius_m(frame.centre_lat) / physics.METRES_PER_KM
        collected["spacecraft_altitude_km"] = distance - radius
    return AcquisitionInfo(**collected)


def _measured_mean(plane: np.ndarray, measured: np.ndarray) -> float | None:
    """Return the mean of one plane over the samples its crop measured.

    Args:
        plane: The quantity at every sample of the crop, over its ground axes.
        measured: Which of those samples are measurements inside the tile's box.

    Returns:
        mean: The mean, or None where no finite sample was measured.
    """
    kept = measured & np.isfinite(plane)
    return float(plane[kept].mean()) if kept.any() else None


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
