"""CRISM over one tile in true colour, as delivered and as the build cleans it."""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np

from analysis.visualization import panels
from analysis.visualization.preprocessing import product
from building.configs import crism as configs
from building.preprocessing.common import cut, geometry
from building.preprocessing.common.models.position import Position
from building.preprocessing.crism.correction import centre_wavelengths, mask
from building.preprocessing.crism.models.observation import (
    LATITUDE_PLANE,
    LONGITUDE_PLANE,
)
from common.models.tile import Tile
from common.pds import images

NAME = "CRISM"
TRU_NM = (600.0, 530.0, 440.0)


def true_colour(cube: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Return a cube's bands nearest 600, 530 and 440 nm as stretched RGB."""
    bands = [int(np.nanargmin(np.abs(centres - nm))) for nm in TRU_NM]
    return np.nan_to_num(np.dstack([panels.stretched(cube[..., at]) for at in bands]))


def delivered_colour(identifier: str, frame: Tile) -> np.ndarray:
    """Return the visible detector's true colour over a tile, as the archive holds it.

    Args:
        identifier: The observation, its files already in the download cache.
        frame: The tile it is cut to.

    Returns:
        colour: Lines by samples by RGB, from 0 to 1.
    """
    image = configs.CACHE.product_files(
        identifier, configs.Kind.OBSERVATION, detector=configs.Detector.VISIBLE
    )[".img"]
    cube, label = images.load_cube(image)
    table = centre_wavelengths.read_calibration(
        configs.WAVELENGTH_FILES[configs.Detector.VISIBLE],
        centre_wavelengths.detector_rows(image, label),
    ).astype("f8")
    backplanes, _ = images.load_cube(
        configs.CACHE.product_files(
            identifier, configs.Kind.GEOMETRY, detector=configs.Detector.INFRARED
        )[".img"]
    )
    lines = min(cube.shape[0], backplanes.shape[0])
    held = cut.overlap(
        Position(
            backplanes[:lines, :, LATITUDE_PLANE],
            backplanes[:lines, :, LONGITUDE_PLANE],
            False,
        ),
        frame,
    )
    floor, ceiling = mask.BRIGHTNESS
    raw = geometry.kept_part(cube[:lines], held.bounds)
    inside = True if held.inside is None else held.inside[..., None]
    raw = np.where(inside & (raw >= floor) & (raw <= ceiling), raw, np.nan)
    return true_colour(raw, centre_wavelengths.band_centres(table))


def plot(identifier: str, frame: Tile) -> widgets.Widget:
    """Show one CRISM observation over a tile in true colour, delivered and built."""
    product.fetch_product(NAME, identifier, frame)
    sample = product.tile_sample(
        NAME, product.product_observation(NAME, identifier), frame
    )
    built = np.where(sample.measured_ground[..., None], sample.cube, np.nan)
    return panels.side_by_side(
        [
            ("Delivered I/F", delivered_colour(identifier, frame)),
            (
                "Built sample",
                true_colour(built, np.asarray(centre_wavelengths.BANDS_NM)),
            ),
        ]
    )
