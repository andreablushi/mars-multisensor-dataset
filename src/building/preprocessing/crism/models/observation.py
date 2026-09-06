"""One CRISM observation as it comes off disk, both detectors as a single cube."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Which DDR backplane places a pixel. The other twelve are dropped: three carry
# the null sentinel in every pixel, four barely vary across a scan, and the rest
# are MOLA resampled onto this grid, which the MOLA tile itself holds better.
BACKPLANES = {"latitude": 3, "longitude": 4}


@dataclass(frozen=True, slots=True)
class CrismObservation:
    """One observation with its two detectors joined.

    Attributes:
        label: What every product it was published as says about it, merged.
        identifier: The observation id.
        cube: Lines by columns by bands, bands ascending in wavelength, holding
            only what both detectors kept.
        wavelengths: The centre wavelength of every column and band, in that
            same order.
        geometry: The backplanes on the same grid, as lines by columns by 14.
        columns: Which of the original 64 samples these columns are.
        valid: Lines by columns, True where the pixel carries a measurement
            rather than a cell the cleaning filled.
    """

    identifier: str
    label: dict[str, str]
    cube: np.ndarray
    wavelengths: np.ndarray
    geometry: np.ndarray
    columns: np.ndarray
    valid: np.ndarray

    # A pushbroom swath bends as the spacecraft flies, so every pixel carries
    # the pair its own backplanes give it.
    separable = False

    @property
    def latitude(self) -> np.ndarray:
        """Return the latitude every pixel was measured at.

        Returns:
            Lines by columns, in degrees.
        """
        return self.geometry[:, :, BACKPLANES["latitude"]]

    @property
    def longitude(self) -> np.ndarray:
        """Return the longitude every pixel was measured at.

        Returns:
            Lines by columns, in degrees.
        """
        return self.geometry[:, :, BACKPLANES["longitude"]]
