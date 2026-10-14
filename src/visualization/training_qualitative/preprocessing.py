"""Each instrument over one tile, as downloaded and as the model receives it."""

from __future__ import annotations

import base64
import shutil
from collections.abc import Callable
from pathlib import Path

import httpx
import ipywidgets as widgets
import numpy as np
import tifffile
from matplotlib.figure import Figure

from analysis import paths
from building.configs import crism, ctx, sharad
from building.models.instrument import INSTRUMENTS
from building.preprocessing.common import cut, geometry
from building.preprocessing.common.models.position import Position
from building.preprocessing.common.models.sample import Sample
from building.preprocessing.crism.correction import centre_wavelengths, mask
from building.preprocessing.crism.models.observation import (
    LATITUDE_PLANE,
    LONGITUDE_PLANE,
)
from building.preprocessing.ctx.crop import reached_lines
from building.preprocessing.ctx.isis import run_isis
from common.disk.files import read_json, write_json
from common.disk.slugify import slugify
from common.fetch.http import TLS_CONTEXT
from common.models.tile import Tile
from common.pds import images
from visualization import panels
from visualization.training_qualitative import config, product

BUILT_NM = np.asarray(centre_wavelengths.BANDS_NM)


def plot(name: str, identifier: str, frame: Tile) -> widgets.VBox:
    """Show one product over a tile, drawn once, then read off disk."""
    path = (
        paths.PANELS_ROOT / slugify(name) / f"{slugify(identifier)}_{frame.name}.json"
    )
    if not path.exists():
        instrument = INSTRUMENTS[name]
        with httpx.Client(verify=TLS_CONTEXT) as client:
            instrument.fetch(identifier, client, (frame,))
        if instrument.place:
            instrument.place(identifier)
        drawn = {
            "CRISM": crism_images,
            "CTX": ctx_images,
            "MOLA": mola_images,
            "SHARAD": sharad_images,
        }
        rows, sample = drawn[name](name, identifier, frame)
        write_json(
            path,
            [
                {"png": base64.b64encode(child.value).decode()}
                if isinstance(child, widgets.Image)
                else {"html": child.value}
                for child in [
                    *rows,
                    product.acquisition_table(name, identifier, sample, frame),
                ]
            ],
        )
    return widgets.VBox(
        [
            panels.png_widget(base64.b64decode(child["png"]))
            if "png" in child
            else widgets.HTML(child["html"])
            for child in read_json(path)
        ]
    )


def composite(
    cube: np.ndarray, centres: np.ndarray, bands_nm: tuple, scale: Callable
) -> np.ndarray:
    """Return a cube's bands nearest three wavelengths as RGB, each band scaled."""
    bands = [int(np.nanargmin(np.abs(centres - nm))) for nm in bands_nm]
    return np.nan_to_num(np.dstack([scale(cube[..., at]) for at in bands]))


def delivered_cube(
    identifier: str, detector: crism.Detector, frame: Tile
) -> tuple[np.ndarray, np.ndarray]:
    """Return one detector's I/F over a tile, as the archive holds it.

    Args:
        identifier: The observation, its files already in the download cache.
        detector: The detector read.
        frame: The tile it is cut to.

    Returns:
        cube: Lines by samples by bands, NaN outside the tile or a reading.
        centres: The centre wavelength of every band in nm.
    """
    image = crism.CACHE.product_files(
        identifier, crism.Kind.OBSERVATION, detector=detector
    )[".img"]
    cube, label = images.load_cube(image)
    table = centre_wavelengths.read_calibration(
        crism.WAVELENGTH_FILES[detector], centre_wavelengths.detector_rows(image, label)
    ).astype("f8")
    backplanes, _ = images.load_cube(
        crism.CACHE.product_files(
            identifier, crism.Kind.GEOMETRY, detector=crism.Detector.INFRARED
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


def crism_images(
    name: str, identifier: str, frame: Tile
) -> tuple[list[widgets.Image], Sample]:
    """Draw one CRISM observation over a tile in true and false colour, every stage.

    Args:
        name: The instrument, as the build names it.
        identifier: The observation, its files already in the download cache.
        frame: The tile it is cut to.

    Returns:
        rows: One row of every stage per view.
        sample: The crop the build would write.
    """
    _, sample = product.observed_sample(name, identifier, frame)
    built, shown, ranged, pooled = (
        np.where(sample.measured_ground[..., None], values, np.nan)
        for values in (
            sample.cube,
            product.model_input(name, sample),
            product.min_max_input(identifier, sample, frame),
            product.min_max_input(identifier, sample, frame, per_band=False),
        )
    )
    delivered = {
        detector: delivered_cube(identifier, detector, frame)
        for detector, _ in config.VIEWS.values()
    }
    rows = [
        product.side_by_side(
            [
                (
                    "Downloaded sample",
                    composite(*delivered[detector], bands, product.stretched),
                ),
                (
                    "Preprocessed sample",
                    composite(built, BUILT_NM, bands, product.stretched),
                ),
                ("Model input", composite(shown, BUILT_NM, bands, np.asarray)),
                ("Min-max input", composite(ranged, BUILT_NM, bands, np.asarray)),
                (
                    "Global min-max input",
                    composite(pooled, BUILT_NM, bands, np.asarray),
                ),
            ],
            label=view,
        )
        for view, (detector, bands) in config.VIEWS.items()
    ]
    return rows, sample


def delivered_counts(cube: Path, lines: tuple[int, int]) -> np.ndarray:
    """Return the raw counts of a placed scan's lines, stretched.

    Args:
        cube: The placed scan, before calibration.
        lines: The first and last line to read, counted from one.

    Returns:
        counts: Lines by samples in camera geometry, from 0 to 1.

    Raises:
        RuntimeError: When an ISIS application fails.
    """
    first, last = lines
    trimmed, image = (cube.with_suffix(suffix) for suffix in (".raw.cub", ".raw.tif"))
    try:
        run_isis(
            "crop",
            {"from": cube, "to": trimmed, "line": first, "nlines": last - first + 1},
        )
        run_isis(
            "isis2std",
            {"from": trimmed, "to": image, "format": "tiff", "bittype": "u16bit"},
        )
        counts = tifffile.imread(image).astype(float)
    finally:
        trimmed.unlink(missing_ok=True)
        image.unlink(missing_ok=True)
    return product.stretched(np.where(counts > 0, counts, np.nan))


def ctx_images(
    name: str, identifier: str, frame: Tile
) -> tuple[list[widgets.Image], Sample]:
    """Draw one CTX scan over a tile, as downloaded and as the model receives it.

    Args:
        name: The instrument, as the build names it.
        identifier: The scan, already fetched and placed.
        frame: The tile it is cut to.

    Returns:
        rows: Both stages, in one row.
        sample: The crop the build would write.
    """
    placed = ctx.CACHE.files(identifier, identifier)[ctx.CUBE_SUFFIX]
    kept = placed.with_suffix(".placed.cub")
    shutil.copyfile(placed, kept)
    try:
        observation, sample = product.observed_sample(name, identifier, frame)
        delivered = delivered_counts(kept, reached_lines(observation, frame))
        observation.cube.unlink()
    finally:
        kept.replace(placed)
    shown = np.where(sample.measured_ground, product.model_input(name, sample), np.nan)
    row = product.side_by_side(
        [("Downloaded sample", delivered), ("Model input", shown)]
    )
    return [row], sample


def mola_images(
    name: str, identifier: str, frame: Tile
) -> tuple[list[widgets.Image], Sample]:
    """Draw one MOLA grid's elevation over a tile, stretched.

    Args:
        name: The instrument, as the build names it.
        identifier: The grid, its sheets already in the download cache.
        frame: The tile it is cut to.

    Returns:
        rows: The elevation, alone in its row.
        sample: The crop the build would write.
    """
    _, sample = product.observed_sample(name, identifier, frame)
    shown = np.where(sample.measured_ground, sample.elevation, np.nan)
    row = product.side_by_side(
        [("Elevation", product.stretched(shown))], cmap=config.ELEVATION_CMAP
    )
    return [row], sample


def sharad_images(
    name: str, identifier: str, frame: Tile
) -> tuple[list[widgets.Image], Sample]:
    """Draw one SHARAD track and its cluttergram whole, beside the model input.

    Args:
        name: The instrument, as the build names it.
        identifier: The track, its files already in the download cache.
        frame: The tile it is cut to.

    Returns:
        rows: The radargram, its cluttergram and the crop, in one figure.
        sample: The crop the build would write.
    """
    observation, sample = product.observed_sample(name, identifier, frame)
    power, _ = images.load_cube(
        sharad.CACHE.product_files(identifier, sharad.Kind.OBSERVATION)[".img"]
    )
    figure = Figure(figsize=(11, 7))
    axes = figure.subplot_mosaic(
        [["radargram", "crop"], ["cluttergram", "crop"]], width_ratios=(2, 1)
    )
    for panel, values, text in (
        ("radargram", product.stretched(power[:, :, 0]), "Downloaded sample"),
        (
            "cluttergram",
            product.stretched(np.asarray(observation.clutter)),
            "Downloaded cluttergram",
        ),
        ("crop", product.model_input(name, sample), "Model input"),
    ):
        axis = axes[panel]
        axis.imshow(
            values, cmap="gray", aspect="auto", interpolation="nearest", vmin=0, vmax=1
        )
        panels.titled(axis, text)
        axis.set_xlabel("Trace")
        axis.set_ylabel("Delay row")
    for panel in ("radargram", "cluttergram"):
        for edge in (sample.traces[0], sample.traces[-1]):
            axes[panel].axvline(edge, color="tab:orange", linewidth=1.0)
    figure.tight_layout()
    return [panels.rendered(figure)], sample
