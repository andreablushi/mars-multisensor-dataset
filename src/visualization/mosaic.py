"""The mosaic of Mars: one crop of it fetched, and drawn under a tile or the planet."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable
from functools import lru_cache

import httpx
import ipywidgets as widgets
import numpy as np
from cartopy import crs
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import imread

from common.fetch.http import TLS_CONTEXT
from common.maths import geodesy
from common.maths.box import POLE, Crop
from common.maths.physics import RADIUS_M
from visualization import config, panels

MARS = Crop(-geodesy.HALF_TURN, -POLE, geodesy.HALF_TURN, POLE)

GLOBE = crs.Globe(semimajor_axis=RADIUS_M, semiminor_axis=RADIUS_M, ellipse=None)
LONLAT = crs.PlateCarree(globe=GLOBE)
ROBINSON = crs.Robinson(globe=GLOBE)

MARS_LAID = dict(
    extent=MARS.extent,
    origin="upper",
    interpolation="nearest",
    transform=LONLAT,
    regrid_shape=(config.MARS_PIXELS, config.MARS_PIXELS // 2),
)


def fetched(
    crop: Crop,
    draw: Callable[[bytes], widgets.Widget],
    pixels: int = config.BASEMAP_PIXELS,
) -> widgets.Box:
    """Claim the space one crop goes in and fill it off the thread that fetches it.

    Args:
        crop: The lon/lat box the crop covers.
        draw: What turns the fetched crop into the figure shown.
        pixels: How many pixels the crop's longer side is fetched at.

    Returns:
        space: A waiting placeholder, replaced by the figure once it is drawn.
    """
    space = widgets.Box(
        [
            widgets.HTML(
                f"<div style='width: 320px; height: 320px;"
                f" display: flex; align-items: center; justify-content: center;"
                f" box-sizing: border-box; padding: 10px; text-align: center;"
                f" background: #f2f2f2; border: 1px solid #d8d8d8;"
                f" border-radius: 4px; color: {config.GREY};"
                f" font-family: sans-serif; font-size: 12px;'>"
                f"Fetching the basemap...</div>"
            )
        ]
    )
    threading.Thread(
        target=_fill, args=(space, crop, draw, pixels), daemon=True
    ).start()
    return space


def _fill(
    space: widgets.Box, crop: Crop, draw: Callable[[bytes], widgets.Widget], pixels: int
) -> None:
    """Fetch one crop and put the figure drawn from it in the claimed space.

    Args:
        space: The space claimed for the figure.
        crop: The lon/lat box the crop covers.
        draw: What turns the fetched crop into the figure shown.
        pixels: How many pixels the crop's longer side is fetched at.
    """
    try:
        image = crop_image(crop, pixels)
    except Exception as exc:
        space.children = (panels.unavailable(config.BASEMAP_FAILED.format(reason=exc)),)
        return
    space.children = (draw(image),)


def read_mosaic(image: bytes) -> np.ndarray:
    """Decode one mosaic crop as fetched, rows from the north."""
    return imread(io.BytesIO(image), format="png")


def mars_board(image: bytes, title: str) -> tuple[Figure, Axes]:
    """Open a figure with the mosaic of Mars drawn under its graticule.

    Args:
        image: The mosaic of the whole planet, as `crop_image` hands it back.
        title: What the map is titled.

    Returns:
        figure: The figure the map is drawn on.
        axis: The map itself, in lon and lat.
    """
    figure = Figure(figsize=config.MARS_FIGURE_SIZE)
    axis = figure.subplots(subplot_kw={"projection": ROBINSON})
    axis.set_global()
    axis.imshow(read_mosaic(image), cmap="gray", **MARS_LAID)
    lines = axis.gridlines(
        LONLAT, draw_labels=True, color=config.GRATICULE, linewidth=0.4, alpha=0.5
    )
    lines.xlabel_style = lines.ylabel_style = {"size": 8}
    panels.titled(axis, title)
    return figure, axis


@lru_cache(maxsize=32)
def crop_image(crop: Crop, pixels: int) -> bytes:
    """Fetch the mosaic over one lon/lat box, held for the panels sharing it.

    Args:
        crop: The lon/lat box to fetch.
        pixels: How many pixels the crop's longer side is fetched at.

    Returns:
        image: The crop, as the PNG the map server sent.

    Raises:
        ValueError: When the server answers with something other than an image.
    """
    tall = crop.north - crop.south
    wide = (crop.east - crop.west) * geodesy.longitude_stretch(crop.centre_lat)
    longest = max(wide, tall)
    response = httpx.get(
        config.BASEMAP_URL,
        verify=TLS_CONTEXT,
        params={
            "map": config.BASEMAP_MAP,
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": config.BASEMAP_LAYER,
            "STYLES": "",
            "SRS": "EPSG:4326",
            "BBOX": ",".join(
                f"{bound:.4f}"
                for bound in (crop.west, crop.south, crop.east, crop.north)
            ),
            "WIDTH": max(1, round(pixels * wide / longest)),
            "HEIGHT": max(1, round(pixels * tall / longest)),
            "FORMAT": "image/png",
        },
        timeout=config.BASEMAP_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()
    if not response.headers.get("content-type", "").startswith("image/"):
        raise ValueError(response.text.strip()[:200])
    return response.content
