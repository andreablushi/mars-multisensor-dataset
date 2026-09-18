"""One CRISM observation as it comes off disk, both detectors as a single cube."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Which DDR backplane places a pixel; the other twelve say nothing MOLA says better.
LATITUDE_PLANE = 3
LONGITUDE_PLANE = 4


@dataclass(frozen=True, slots=True)
class CrismObservation:
    """One observation with its two detectors joined.

    Attributes:
        label: What every product it was published as says about it, merged.
        identifier: The observation id.
        cube: Lines by columns by the whole of the survey's band grid, holding only
            the columns both detectors kept and NaN for the bands it never measured.
        geometry: The backplanes on the same grid, as lines by columns by 14.
        valid: Lines by columns, True where the pixel carries a measurement
            rather than a cell the cleaning filled.
        measured_bands: One flag per band of that grid, True where this observation
            measured it, so an empty band is told from a refused pixel.
    """

    identifier: str
    label: dict[str, str]
    cube: np.ndarray
    geometry: np.ndarray
    valid: np.ndarray
    measured_bands: np.ndarray

    # A pushbroom swath bends, so every pixel carries its own backplanes' pair.
    separable = False

    @property
    def latitude(self) -> np.ndarray:
        """Return the latitude every pixel was measured at.

        Returns:
            latitude: Lines by columns, in degrees.
        """
        return self.geometry[:, :, LATITUDE_PLANE]

    @property
    def longitude(self) -> np.ndarray:
        """Return the longitude every pixel was measured at.

        Returns:
            longitude: Lines by columns, in degrees.
        """
        return self.geometry[:, :, LONGITUDE_PLANE]
