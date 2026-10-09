"""Removing what the Martian atmosphere adds to a scan, rather than the ground."""

from __future__ import annotations

import numpy as np

from building.configs import crism as configs
from building.preprocessing.crism.correction import centre_wavelengths


def read_transmission(label: dict[str, str], rows: np.ndarray) -> np.ndarray:
    """Read the atmosphere's transmission used by the volcano scan.

    Args:
        label: The scan's label, which says when it started and so picks the file.
        rows: The detector row of every band of the scan, in stored order.

    Returns:
        transmission: Per column and band in the order of the rows, NaN where unknown.

    Raises:
        ValueError: When the scan started before every period.
    """
    clock = float(label["SPACECRAFT_CLOCK_START_COUNT"].split("/")[1])
    records = sorted(configs.CALIBRATION_ROOT.glob("cdr*_at*.img"))
    started = [one for one in records if int(one.name[5:15]) <= clock]
    if not started:
        raise ValueError(f"No transmission record covers a scan started at {clock}.")
    return centre_wavelengths.read_calibration(started[-1], rows)


def co2_depth(
    cube: np.ndarray,
    refused: np.ndarray,
    transmission: np.ndarray,
    centres: np.ndarray,
) -> np.ndarray:
    """Return each pixel's CO2 depth, the exponent its transmission is raised to.

    Args:
        cube: The values as lines by samples by bands.
        refused: Lines by samples, True where the pixel is not a measurement.
        transmission: The atmosphere's transmission per sample and band, NaN unknown.
        centres: The centre wavelength of every band.

    Returns:
        depth: Lines by samples, a pixel lost in its noise taking the strip's own.
    """
    near, far = (int(np.argmin(np.abs(centres - nm))) for nm in configs.CO2_BAND_NM)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.log(cube[:, :, near] / cube[:, :, far]) / np.log(
            transmission[:, near] / transmission[:, far]
        )
    read = ~refused & np.isfinite(depth) & (depth > 0)
    # A pixel whose depth is lost in its noise takes the strip's own.
    depth[~read] = np.median(depth[read]) if read.any() else 1.0
    return depth


def remove_co2_bands(
    cube: np.ndarray, table: np.ndarray, transmission: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop the bands inside the CO2 window, whose depth the atmosphere sets.

    Args:
        cube: The values as lines by samples by bands.
        table: The centre wavelength of every column and band.
        transmission: The atmosphere's transmission per sample and band.

    Returns:
        cube: The cube without those bands.
        table: The wavelengths without those bands.
        transmission: The transmission without those bands.
    """
    dry = centre_wavelengths.outside_co2(centre_wavelengths.band_centres(table))
    return np.ascontiguousarray(cube[:, :, dry]), table[:, dry], transmission[:, dry]


def divide_transmission(
    cube: np.ndarray, transmission: np.ndarray, depth: np.ndarray
) -> None:
    """Divide each pixel by the transmission raised to its own CO2 depth, as CAT does.

    Args:
        cube: The values as lines by samples by bands, divided in place.
        transmission: The atmosphere's transmission per sample and band, NaN unknown.
        depth: Lines by samples, the exponent of each pixel.
    """
    np.divide(
        cube,
        transmission ** depth[:, :, None],
        out=cube,
        where=np.isfinite(transmission),
    )
