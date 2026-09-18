"""One CRISM observation cut to the tile it was kept for."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from building.preprocessing.common.models.sample import Sample


@dataclass(frozen=True, slots=True, kw_only=True)
class CrismSample(Sample):
    """The spectra one observation measured over one tile.

    Attributes:
        cube: Lines by columns by the whole of the survey's band grid, NaN for the
            bands the observation never measured.
        incidence_deg: The angle between the Sun and the areoid's normal at every
            pixel, as the DDR wrote it.
        emission_deg: The angle between the spacecraft and that normal, the same
            way.
        phase_deg: The angle the ground sees between the two, the same way.
        local_solar_time_h: The hour of the Martian day at every pixel.
    """

    cube: np.ndarray
    incidence_deg: np.ndarray
    emission_deg: np.ndarray
    phase_deg: np.ndarray
    local_solar_time_h: np.ndarray
