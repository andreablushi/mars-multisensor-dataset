"""What every instrument's crop holds, whatever its own arrays are."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.position import Position


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    """One observation cut to its tile, in the shape its instrument publishes.

    Attributes:
        position: Where the remaining samples sit, in degrees or projected metres.
        label: What every product of the observation says about it, merged.
        inside: Which of them truly falls in the tile's box, or None for all.
        valid: Which of them is a measurement rather than a fill, or None for all.
        measured_bands: One flag per band this crop measured, or None without bands.
        incidence_deg: The Sun's angle from the areoid normal per sample, or None.
        emission_deg: The spacecraft's angle from that normal, or None.
        phase_deg: The angle the ground sees between the two, or None likewise.
        local_solar_time_h: The local Martian hour at every sample, or None.
        spacecraft_altitude_km: The spacecraft's height per sample, or None.
    """

    position: Position
    label: dict[str, str]
    inside: np.ndarray | None = None
    valid: np.ndarray | None = None
    measured_bands: np.ndarray | None = None
    incidence_deg: np.ndarray | None = None
    emission_deg: np.ndarray | None = None
    phase_deg: np.ndarray | None = None
    local_solar_time_h: np.ndarray | None = None
    spacecraft_altitude_km: np.ndarray | None = None

    @property
    def measured_ground(self) -> np.ndarray:
        """Return which samples are both in the tile's box and measurements.

        Returns:
            measured_ground: One flag per sample over the ground axes, both masks.
        """
        held = np.ones(self.position.sizes, dtype=bool)
        for mask in (self.inside, self.valid):
            if mask is not None:
                held = held & mask
        return held
