"""The SWIM ice consistency maps: fetched once, and read back from their archives."""

from __future__ import annotations

import io
import zipfile

import httpx
import numpy as np
import tifffile

from mars_multisensor_dataset.analysis import paths
from mars_multisensor_dataset.common.fetch.http import (
    REQUEST_TIMEOUT,
    TLS_CONTEXT,
    FetchError,
    streamed,
)
from mars_multisensor_dataset.common.maths import geodesy

LAYERS = {
    "Ci_1_5": (paths.SWIM_MAPS_PATH, "SWIM4MIM_Ci_1_5.tif"),
    "Ci_5": (paths.SWIM_MAPS_PATH, "SWIM4MIM_Ci_5.tif"),
    "RS": (paths.SWIM_RADAR_PATH, "SWIM2_RS.tif"),
}

ARCHIVES = {
    paths.SWIM_MAPS_PATH: (
        "https://swim.psi.edu/SWIM4MIMProducts.php",
        ("Ci_1_5-GeoTIFF", "Ci_5-GeoTIFF"),
    ),
    paths.SWIM_RADAR_PATH: (
        "https://swim.psi.edu/SWIM2Products.php",
        ("C_RS-GeoTIFF",),
    ),
}

# Every map is simple cylindrical at 20 pixels a degree, from 180W and 60N.
PIXELS_PER_DEGREE = 20
WEST_LON = -180.0
NORTH_LAT = 60.0


def download_maps(force: bool) -> None:
    """Fetch every archive the layers read come in, unless it is already on disk.

    Args:
        force: Whether to fetch them again even when they are on disk.

    Raises:
        FetchError: When a products page answers without the archive it builds.
    """
    with httpx.Client(verify=TLS_CONTEXT, timeout=REQUEST_TIMEOUT) as client:
        for path, (url, fields) in ARCHIVES.items():
            if path.exists() and not force:
                continue
            # The page zips the layers asked for and redirects to where it put them
            asked = client.get(
                url, params=dict.fromkeys(fields, "True") | {"DL-Sel": "True"}
            )
            if asked.next_request is None:
                raise FetchError(f"{url} answered {asked.status_code}, no archive")
            streamed(str(asked.next_request.url), path, client=client, spans=())


def read_consistency_map(layer: str) -> np.ndarray:
    """Read one layer from the archive it comes in.

    Args:
        layer: The layer, as SWIM names it, such as "Ci_1_5".

    Returns:
        consistency: Its ice consistency, rows from 60N and columns from 180W. (R, C)
    """
    path, name = LAYERS[layer]
    with zipfile.ZipFile(path) as archive:
        return tifffile.imread(io.BytesIO(archive.read(name))).astype(np.float32)


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
