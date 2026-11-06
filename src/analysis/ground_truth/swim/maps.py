"""The SWIM ice consistency maps: fetched once, and read back from their archive."""

from __future__ import annotations

import io
import zipfile

import httpx
import numpy as np
import tifffile

from analysis import paths
from common.fetch.http import REQUEST_TIMEOUT, TLS_CONTEXT, FetchError, streamed
from common.maths import geodesy

PRODUCTS_URL = "https://swim.psi.edu/SWIM4MIMProducts.php"

LAYERS = ("Ci_1_5", "Ci_5")

# Every map is simple cylindrical at 20 pixels a degree, from 180W and 60N.
PIXELS_PER_DEGREE = 20
WEST_LON = -180.0
NORTH_LAT = 60.0


def download_maps(force: bool) -> None:
    """Fetch the archive of every layer read, unless it is already on disk.

    Args:
        force: Whether to fetch it again even when it is on disk.

    Raises:
        FetchError: When the products page answers without the archive it builds.
    """
    if paths.SWIM_MAPS_PATH.exists() and not force:
        return
    with httpx.Client(verify=TLS_CONTEXT, timeout=REQUEST_TIMEOUT) as client:
        # The page zips the layers asked for and redirects to where it put them
        asked = client.get(
            PRODUCTS_URL,
            params={f"{layer}-GeoTIFF": "True" for layer in LAYERS}
            | {"DL-Sel": "True"},
        )
        if asked.next_request is None:
            raise FetchError(f"{PRODUCTS_URL} answered {asked.status_code}, no archive")
        streamed(
            str(asked.next_request.url), paths.SWIM_MAPS_PATH, client=client, spans=()
        )


def read_consistency_map(layer: str) -> np.ndarray:
    """Read one layer from the archive.

    Args:
        layer: The layer, as SWIM names it, such as "Ci_1_5".

    Returns:
        consistency: Its ice consistency, rows from 60N and columns from 180W. (R, C)
    """
    with zipfile.ZipFile(paths.SWIM_MAPS_PATH) as archive:
        return tifffile.imread(io.BytesIO(archive.read(f"SWIM4MIM_{layer}.tif")))


def map_pixels(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the pixel every point falls in on any of the maps.

    Args:
        lon: The longitudes of the points, in degrees.
        lat: The latitudes of the points, in degrees.

    Returns:
        rows: The row each point falls in.
        columns: The column each point falls in, wrapped across 180 degrees.
    """
    rows = np.floor((NORTH_LAT - lat) * PIXELS_PER_DEGREE).astype(int)
    columns = np.floor(
        (geodesy.normalise_longitude(lon) - WEST_LON) * PIXELS_PER_DEGREE
    ).astype(int)
    return (
        np.clip(rows, 0, round(2 * NORTH_LAT * PIXELS_PER_DEGREE) - 1),
        columns % round(geodesy.TURN * PIXELS_PER_DEGREE),
    )
