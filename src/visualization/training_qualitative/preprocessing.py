"""Each instrument over one tile, as downloaded and as the model receives it."""

from __future__ import annotations

import base64
import shutil
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import httpx
import ipywidgets as widgets
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analysis import paths
from analysis.utils.tile_group import tile_grid
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
from building.preprocessing.ctx.crop import projected_pixels
from building.preprocessing.ctx.isis import export_image
from building.preprocessing.ctx.models.sample import BLANK
from building.preprocessing.sharad.normalize import (
    clutter_db,
    radargram_db,
    surface_aligned,
)
from building.preprocessing.sharad.read import clutter_power
from common.disk.files import read_json, write_json
from common.disk.slugify import slugify
from common.fetch.http import TLS_CONTEXT
from common.models.tile import Tile
from common.pds import images
from visualization import panels
from visualization.training_qualitative import config, product

BUILT_NM = np.asarray(centre_wavelengths.BANDS_NM)

RAW_EXPORT = partial(export_image, span=config.RAW_COUNTS)


def plot(name: str, frame: Tile, identifier: str | None = None) -> widgets.VBox:
    """Show every product the index holds of a tile as one mosaic, or one product."""
    identifiers = (
        (identifier,) if identifier else product.indexed_identifiers(name, frame)
    )
    path = (
        paths.PANELS_ROOT
        / slugify(name)
        / f"{slugify(identifier or config.MOSAIC)}_{frame.name}.json"
    )
    drawn = {"CRISM": crism_images, "CTX": ctx_images, "MOLA": mola_images}
    return cached_panels(path, name, identifiers, frame, drawn[name])


def plot_sharad(clip: tuple[float, float] = config.SHARAD_CLIP) -> widgets.VBox:
    """Show each hardcoded SHARAD track whole, cropped, and as the model reads it."""
    grid = tile_grid()
    low, high = clip
    return widgets.VBox(
        [
            cached_panels(
                paths.PANELS_ROOT
                / slugify("SHARAD")
                / f"{identifier}_{tile}_clip_{low:g}_{high:g}.json",
                "SHARAD",
                (identifier,),
                grid.tile_named(tile),
                partial(sharad_images, title=title, clip=clip),
            )
            for title, (identifier, tile) in config.SHARAD_SAMPLES.items()
        ]
    )


def cached_panels(
    path: Path,
    name: str,
    identifiers: Sequence[str],
    frame: Tile,
    draw: Callable[..., tuple[list[widgets.Widget], list[Sample]]],
) -> widgets.VBox:
    """Return the panels of products over a tile, drawn once and kept at a path.

    Args:
        path: Where the drawn panels are kept.
        name: The instrument, as the build names it.
        identifiers: The products, fetched first when not drawn yet.
        frame: The tile they are cut to.
        draw: What draws the products' rows and the crops the build would write.

    Returns:
        panels: The rows, then what every crop was taken under.
    """
    if not path.exists():
        instrument = INSTRUMENTS[name]
        with httpx.Client(verify=TLS_CONTEXT) as client:
            for one in identifiers:
                instrument.fetch(one, client, (frame,))
                if instrument.place:
                    instrument.place(one)
        rows, samples = draw(name, identifiers, frame)
        write_json(
            path,
            [
                {"png": base64.b64encode(child.value).decode()}
                if isinstance(child, widgets.Image)
                else {"html": child.value}
                for child in [
                    *rows,
                    product.acquisition_table(name, identifiers, samples, frame),
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


def composite_layers(
    layers: Sequence[tuple[Position, np.ndarray, np.ndarray]],
    bands_nm: tuple,
    stretch: bool,
) -> list[product.Layer]:
    """Return each crop's bands nearest three wavelengths as RGB, stretched alike.

    Args:
        layers: Each crop's position, its cube as lines by samples by bands, and
            the centre of every band in nm.
        bands_nm: The wavelengths drawn as red, green and blue.
        stretch: Whether each colour is stretched across every crop at once.

    Returns:
        layers: Each crop's position beside its RGB values.
    """
    picked = [
        cube[..., [int(np.nanargmin(np.abs(centres - nm))) for nm in bands_nm]]
        for _, cube, centres in layers
    ]
    if stretch:
        colours = [
            product.stretched_alike([one[..., at] for one in picked])
            for at in range(len(bands_nm))
        ]
        picked = [np.dstack(planes) for planes in zip(*colours, strict=True)]
    return [
        (position, values)
        for (position, _, _), values in zip(layers, picked, strict=True)
    ]


def delivered_cube(
    identifier: str, detector: crism.Detector, frame: Tile
) -> tuple[Position, np.ndarray, np.ndarray]:
    """Return one detector's I/F over a tile, as the archive holds it.

    Args:
        identifier: The observation, its files already in the download cache.
        detector: The detector read.
        frame: The tile it is cut to.

    Returns:
        position: Where the kept samples sit.
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
    return held.position, raw, centre_wavelengths.band_centres(table)


def crism_images(
    name: str, identifiers: Sequence[str], frame: Tile
) -> tuple[list[widgets.Image], list[Sample]]:
    """Draw a tile's CRISM observations in true and false colour, every stage.

    Args:
        name: The instrument, as the build names it.
        identifiers: The observations, oldest first, their files in the cache.
        frame: The tile they are cut to.

    Returns:
        rows: One row per stage, its mosaic in every view side by side.
        samples: The crops the build would write, one per observation.
    """
    samples = []
    built = {title: [] for title in config.BUILT_STAGES}
    for identifier in identifiers:
        _, sample = product.observed_sample(name, identifier, frame)
        samples.append(sample)
        stages = (
            sample.cube,
            product.model_input(name, sample),
            product.min_max_input(identifier, sample, frame),
            product.min_max_input(identifier, sample, frame, per_band=False),
        )
        for layers, values in zip(built.values(), stages, strict=True):
            held = np.where(sample.measured_ground[..., None], values, np.nan)
            layers.append((sample.position, held, BUILT_NM))
    delivered = {
        detector: [
            delivered_cube(identifier, detector, frame) for identifier in identifiers
        ]
        for detector, _ in config.VIEWS.values()
    }
    rows = [
        product.side_by_side(
            [
                (
                    view,
                    composite_layers(
                        delivered[detector]
                        if title == config.DOWNLOADED
                        else built[title],
                        bands,
                        title in config.STRETCHED,
                    ),
                )
                for view, (detector, bands) in config.VIEWS.items()
            ],
            label=title,
            height=config.CRISM_ROW_HEIGHT,
        )
        for title in (config.DOWNLOADED, *built)
    ]
    return rows, samples


def ctx_images(
    name: str, identifiers: Sequence[str], frame: Tile
) -> tuple[list[widgets.Image], list[Sample]]:
    """Draw a tile's CTX scans, as downloaded and as the model receives them.

    Args:
        name: The instrument, as the build names it.
        identifiers: The scans, oldest first, already fetched and placed.
        frame: The tile they are cut to.

    Returns:
        rows: Both stages' mosaics, in one row.
        samples: The crops the build would write, one per scan.
    """
    samples, delivered, shown = [], [], []
    for identifier in identifiers:
        placed = ctx.CACHE.files(identifier, identifier)[ctx.CUBE_SUFFIX]
        kept = placed.with_suffix(".placed.cub")
        shutil.copyfile(placed, kept)
        try:
            observation, sample = product.observed_sample(name, identifier, frame)
            pixels, held, _ = projected_pixels(observation, kept, frame, RAW_EXPORT)
            observation.cube.unlink()
        finally:
            kept.replace(placed)
        samples.append(sample)
        inside = True if held.inside is None else held.inside
        counts = np.where(inside & (pixels != BLANK), pixels, np.nan)
        delivered.append((held.position, counts))
        values = product.model_input(name, sample)
        shown.append(
            (sample.position, np.where(sample.measured_ground, values, np.nan))
        )
    stretched = product.stretched_alike([counts for _, counts in delivered])
    row = product.side_by_side(
        [
            (
                config.DOWNLOADED,
                [
                    (position, counts)
                    for (position, _), counts in zip(delivered, stretched, strict=True)
                ],
            ),
            ("Model input", shown),
        ]
    )
    return [row], samples


def mola_images(
    name: str, identifiers: Sequence[str], frame: Tile
) -> tuple[list[widgets.Image], list[Sample]]:
    """Draw a tile's MOLA grids' elevation, stretched alike.

    Args:
        name: The instrument, as the build names it.
        identifiers: The grids, their sheets already in the download cache.
        frame: The tile they are cut to.

    Returns:
        rows: The elevation's mosaic, alone in its row.
        samples: The crops the build would write, one per grid.
    """
    samples = [product.observed_sample(name, one, frame)[1] for one in identifiers]
    elevations = product.stretched_alike(
        [np.where(one.measured_ground, one.elevation, np.nan) for one in samples]
    )
    row = product.side_by_side(
        [
            (
                "Elevation",
                [
                    (one.position, elevation)
                    for one, elevation in zip(samples, elevations, strict=True)
                ],
            )
        ],
        cmap=config.ELEVATION_CMAP,
    )
    return [row], samples


def shown_range(values: np.ndarray) -> tuple[float, float]:
    """Return the 2nd percentile and the maximum of an array's finite values."""
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, config.STRETCH[0])), float(finite.max())


def shown_crop(
    figure: Figure,
    axis: Axes,
    values: np.ndarray,
    edges: tuple[float, float],
    text: str,
) -> None:
    """Draw one crop in grey between two edges, titled, its colour bar beside it."""
    image = axis.imshow(
        values,
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
        vmin=edges[0],
        vmax=edges[1],
    )
    panels.titled(axis, text)
    figure.colorbar(image, ax=axis)


def sharad_ranges(
    cropped: tuple[np.ndarray, np.ndarray],
    offsets: tuple[float, float],
    models: tuple[np.ndarray, np.ndarray],
) -> widgets.HTML:
    """Tabulate each array's range in dB, its surface offset, then its model range."""
    edges = (np.nanmin, np.nanmax)
    rows = [
        (
            kind,
            *(f"{edge(crop):.1f}" for edge in edges),
            f"{offset:.1f}",
            *(f"{edge(model):.2f}" for edge in edges),
        )
        for kind, crop, offset, model in zip(
            config.SHARAD_KINDS, cropped, offsets, models, strict=True
        )
    ]
    return panels.written("Min-max", config.RANGES, rows)


def sharad_images(
    name: str,
    identifiers: Sequence[str],
    frame: Tile,
    title: str,
    clip: tuple[float, float],
) -> tuple[list[widgets.Widget], list[Sample]]:
    """Draw one SHARAD track whole, then its crop in dB, aligned, and as read.

    Args:
        name: The instrument, as the build names it.
        identifiers: The track alone, its files already in the download cache.
        frame: The tile it is cut to.
        title: What the track shows, written above it.
        clip: The low and high its standardised arrays are clipped to.

    Returns:
        rows: The radargram and its cluttergram whole, cropped in dB, each aligned
            on its own surface, then both standardised alike and clipped, in one
            figure, then their ranges.
        samples: The crop the build would write.
    """
    (identifier,) = identifiers
    _, sample = product.observed_sample(name, identifier, frame)
    power, _ = images.load_cube(
        sharad.CACHE.product_files(identifier, sharad.Kind.OBSERVATION)[".img"]
    )
    power = power[:, :, 0]
    clutter = clutter_power(
        sharad.CACHE.product_files(identifier, sharad.Kind.CLUTTER)[".img"],
        power.shape,
    )
    cropped = tuple(
        np.where(sample.valid, values[:, sample.traces], np.nan)
        for values in (radargram_db(power), clutter_db(clutter))
    )
    aligned = tuple(surface_aligned(crop) for crop in cropped)
    # What the alignment subtracted, the same on every value of a crop
    offsets = tuple(
        float(np.nanmean(crop - one))
        for crop, one in zip(cropped, aligned, strict=True)
    )
    models = tuple(product.sharad_model_input(one, clip) for one in aligned)
    low, high = clip
    columns = {
        "Cropped sample (dB)": (cropped, None),
        "Aligned on its surface (dB)": (
            aligned,
            shown_range(np.concatenate([one.ravel() for one in aligned])),
        ),
        f"Model input (z), clipped to [{low:g}, {high:g}]": (models, clip),
    }
    figure = Figure(figsize=(24, 7))
    figure.suptitle(f"{title}: {identifier} on {frame.name}", fontsize=14)
    axes = figure.subplots(2, 4, width_ratios=(2, 1, 1, 1))
    fetched = power.any(axis=0)
    kinds = config.SHARAD_KINDS
    for axis, whole, kind in zip(axes[:, 0], (power, clutter), kinds, strict=True):
        axis.imshow(
            product.stretched(np.where(fetched, whole, np.nan)),
            cmap="gray",
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        panels.titled(axis, f"Downloaded sample\n{kind}")
        for edge in (sample.traces[0], sample.traces[-1]):
            axis.axvline(edge, color="tab:orange", linewidth=1.0)
    for column, (text, (pair, shared)) in enumerate(columns.items(), start=1):
        for axis, crop, kind in zip(axes[:, column], pair, kinds, strict=True):
            shown_crop(
                figure, axis, crop, shared or shown_range(crop), f"{text}\n{kind}"
            )
    for axis in axes.flat:
        axis.set_xlabel("Trace")
        axis.set_ylabel("Delay row")
    figure.tight_layout()
    return [panels.rendered(figure), sharad_ranges(cropped, offsets, models)], [sample]
