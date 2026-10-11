"""CRISM over one tile in true and false colour, as delivered and as the model sees."""

from __future__ import annotations

from collections.abc import Callable

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
BUILT_NM = np.asarray(centre_wavelengths.BANDS_NM)
VIEWS = {
    "True colour R600 G530 B440": (configs.Detector.VISIBLE, (600.0, 530.0, 440.0)),
    "False colour R2529 G1506 B1080": (
        configs.Detector.INFRARED,
        (2529.0, 1506.0, 1080.0),
    ),
}


Delivered = tuple[np.ndarray, np.ndarray]


def composite(
    cube: np.ndarray, centres: np.ndarray, bands_nm: tuple, scale: Callable
) -> np.ndarray:
    """Return a cube's bands nearest three wavelengths as RGB, each band scaled."""
    bands = [int(np.nanargmin(np.abs(centres - nm))) for nm in bands_nm]
    return np.nan_to_num(np.dstack([scale(cube[..., at]) for at in bands]))


def delivered_cube(
    identifier: str, detector: configs.Detector, frame: Tile
) -> Delivered:
    """Return one detector's I/F over a tile, as the archive holds it.

    Args:
        identifier: The observation, its files already in the download cache.
        detector: The detector read.
        frame: The tile it is cut to.

    Returns:
        cube: Lines by samples by bands, NaN outside the tile or a reading.
        centres: The centre wavelength of every band in nm.
    """
    image = configs.CACHE.product_files(
        identifier, configs.Kind.OBSERVATION, detector=detector
    )[".img"]
    cube, label = images.load_cube(image)
    table = centre_wavelengths.read_calibration(
        configs.WAVELENGTH_FILES[detector],
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
    return raw, centre_wavelengths.band_centres(table)


def drawn_panel(identifier: str, frame: Tile) -> widgets.VBox:
    """Show one CRISM observation over a tile in true and false colour, every stage."""
    product.fetch_product(NAME, identifier, frame)
    sample = product.tile_sample(
        NAME, product.product_observation(NAME, identifier), frame
    )
    built, shown = (
        np.where(sample.measured_ground[..., None], values, np.nan)
        for values in (sample.cube, product.model_input(NAME, sample))
    )
    delivered = {
        detector: delivered_cube(identifier, detector, frame)
        for detector, _ in VIEWS.values()
    }
    rows = [
        panels.side_by_side(
            [
                (
                    "Downloaded sample",
                    composite(*delivered[detector], bands, panels.stretched),
                ),
                (
                    "Preprocessed sample",
                    composite(built, BUILT_NM, bands, panels.stretched),
                ),
                (
                    "Model input",
                    composite(shown, BUILT_NM, bands, product.scaled_input),
                ),
            ],
            label=view,
        )
        for view, (detector, bands) in VIEWS.items()
    ]
    return widgets.VBox(
        [*rows, product.acquisition_table(NAME, identifier, sample, frame)]
    )


def plot(identifier: str, frame: Tile) -> widgets.VBox:
    """Show one CRISM observation, drawn once, then read off disk."""
    return product.kept_panel(NAME, identifier, frame, drawn_panel)
