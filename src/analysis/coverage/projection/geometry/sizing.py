"""How big an observation is on the ground: its swath, and its pixel."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from shapely import from_wkt

from analysis.coverage.projection.geometry import footprints
from analysis.models.observation import Observation
from shared.maths import geodesy, physics

# SHARAD transmits 15-25 MHz; its centre sets the sounding wavelength
SHARAD_CENTRE_FREQUENCY_HZ = 20e6
SHARAD_WAVELENGTH_M = physics.SPEED_OF_LIGHT_M_S / SHARAD_CENTRE_FREQUENCY_HZ

# A sounding is as wide as its swath and as long as a spacing ODE never publishes
SHARAD_ALONG_TRACK_M = 460.0

# Ground pixel size in metres for the sets ODE publishes no map scale for
FALLBACK_PIXEL_M = {"MRO/CRISM/TRDR:msp*if*trr3": 180.0, "MRO/CTX/EDR": 5.4}


def track_widths(observations: Sequence[Observation]) -> list[float | None]:
    """Derive a swath width for every ground track among the observations.

    Args:
        observations: The observations to inspect.

    Returns:
        widths: One width in metres per observation, None where the footprint has area.
    """
    widths: list[float | None] = [None] * len(observations)
    for position, observation in enumerate(observations):
        if not observation.is_track or observation.duration_s <= 0.0:
            continue
        # A track is published as lines, whose ground lengths add up to its own
        parts, _ = footprints.single_parts(
            np.asarray([from_wkt(observation.wkt)], dtype=object)
        )
        length = sum(
            geodesy.haversine_length(*np.asarray(part.coords).T) for part in parts
        )
        # The speed that trace implies fixes the altitude, and so the swath
        speed = length / observation.duration_s
        radius = (physics.MARS_GM * physics.RADIUS_M**2 / speed**2) ** (1.0 / 3.0)
        altitude = radius - physics.RADIUS_M
        widths[position] = 2.0 * math.sqrt(SHARAD_WAVELENGTH_M * altitude / 2.0)
    return widths


def ground_pixel_km2(
    set_key: str, map_scale_m: float | None, width_km: float | None
) -> float:
    """Return the ground one pixel of an observation covers.

    Args:
        set_key: The instrument set the observation was asked for by.
        map_scale_m: The ground size of one pixel, or None to use the configured one.
        width_km: The swath width, set only for a sounder's track.

    Returns:
        km2: The ground one pixel covers in square kilometres.

    Raises:
        KeyError: When a set publishes no scale and none is configured for it.
    """
    if width_km is not None:
        return width_km * SHARAD_ALONG_TRACK_M / 1000.0
    scale = map_scale_m or FALLBACK_PIXEL_M.get(set_key)
    if scale is None:
        raise KeyError(
            f"{set_key} publishes no map scale and none is configured for it, "
            f"so it needs an entry in FALLBACK_PIXEL_M spelled exactly this way"
        )
    return (scale / 1000.0) ** 2
