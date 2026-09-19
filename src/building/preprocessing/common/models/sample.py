"""What every instrument's crop holds, whatever its own arrays are."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.relative_position import (
    RelativePosition,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    """One observation cut to its tile, in the shape its instrument publishes.

    Attributes:
        identifier: What the instrument was asked for, its observation or sheet.
        position: Where the samples that are left sit, in degrees from the
            tile's own centre or in the metres of the projection they
            were placed on.
        label: What every product the observation was published as says about
            it, merged into one.
        inside: Which of them truly falls in the tile's box, or None where
            every one of them does.
        valid: Which of them is a measurement rather than a filled cell, or
            None where every one of them is.
        measured_bands: One flag per band of the grid a spectral instrument's crops
            are laid out on, True where this one measured it, or None where the
            instrument holds no band.
        incidence_deg: The angle between the Sun and the areoid's normal at every
            sample, or None where the archive publishes none per sample. A
            sounder's solar zenith angle is that same angle and is carried as it.
        emission_deg: The angle between the spacecraft and that normal, read the
            same way, or None for the same reason.
        phase_deg: The angle the ground sees between the two, or None likewise.
        local_solar_time_h: The hour of the Martian day at every sample, on the
            twenty four hour clock, or None likewise.
        spacecraft_altitude_km: How far above the ground every sample was taken
            from, or None likewise.
    """

    identifier: str
    position: RelativePosition
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
            measured_ground: One flag per sample over the ground axes, the two masks
                rooted together and every sample marked where neither says less.
        """
        held = np.ones(self.position.ground_sizes, dtype=bool)
        for mask in (self.inside, self.valid):
            if mask is not None:
                held = held & mask
        return held
