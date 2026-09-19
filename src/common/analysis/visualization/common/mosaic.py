"""The mosaic under a tile, or under Mars: fetching one crop of it, and drawing it."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable
from functools import lru_cache
from html import escape

import httpx
import ipywidgets as widgets
import numpy as np
from matplotlib import image as reading
from matplotlib.axes import Axes

from common.analysis.visualization.common import panels
from common.analysis.visualization.common.models.box import Box
from common.maths import geodesy

BASEMAP_URL = "https://planetarymaps.usgs.gov/cgi-bin/mapserv"
BASEMAP_MAP = "/maps/mars/mars_simp_cyl.map"
BASEMAP_LAYER = "THEMIS"
BASEMAP_PIXELS = 900
BASEMAP_TIMEOUT = 30.0
BASEMAP_FAILED = "The basemap could not be fetched: {reason}"
BASEMAP_LOADING = "Fetching the basemap..."
BASEMAP_CACHE = 32

PLACEHOLDER = "320px"

NO_BOX = "this tile has no lon/lat box to crop the mosaic to"


def fetched(
    box: Box, draw: Callable[[bytes], widgets.Widget], pixels: int = BASEMAP_PIXELS
) -> widgets.Box:
    """Claim the space one crop goes in and fill it off the thread that fetches it."""
    space = widgets.Box(
        [
            widgets.HTML(
                f"<div style='width: {PLACEHOLDER}; height: {PLACEHOLDER};"
                f" display: flex; align-items: center; justify-content: center;"
                f" box-sizing: border-box; padding: 10px; text-align: center;"
                f" background: #f2f2f2; border: 1px solid #d8d8d8;"
                f" border-radius: 4px; color: {panels.GREY};"
                f" font-family: sans-serif; font-size: 12px;'>"
                f"{escape(BASEMAP_LOADING)}</div>"
            )
        ]
    )

    def fill() -> None:
        """Crop the mosaic and put what it draws in the claimed space."""
        try:
            image = crop(box, pixels)
        except Exception as exc:
            space.children = (panels.unavailable(BASEMAP_FAILED.format(reason=exc)),)
            return
        space.children = (draw(image),)

    threading.Thread(target=fill, daemon=True).start()
    return space


def read_mosaic(image: bytes) -> np.ndarray:
    """Decode one mosaic crop as fetched.

    Args:
        image: The crop, as `crop` hands it back.

    Returns:
        pixels: Its pixels, rows from the north.
    """
    return reading.imread(io.BytesIO(image), format="png")


def draw(axis: Axes, box: Box, image: bytes) -> None:
    """Draw one mosaic crop onto an axis, labelled in lon and lat."""
    axis.imshow(
        read_mosaic(image),
        extent=box.extent,
        origin="upper",
        cmap="gray",
    )
    axis.set_aspect(1.0 / geodesy.longitude_stretch(box.centre_lat))
    axis.set_xlim(box.west, box.east)
    axis.set_ylim(box.south, box.north)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.tick_params(labelsize=8)
    # A footprint reaching well past the crop is cut to it rather than framed
    axis.autoscale(False)


@lru_cache(maxsize=BASEMAP_CACHE)
def crop(box: Box, pixels: int = BASEMAP_PIXELS) -> bytes:
    """Fetch the mosaic over one lon/lat box, held for the panels sharing it."""
    tall = box.north - box.south
    wide = (box.east - box.west) * geodesy.longitude_stretch(box.centre_lat)
    longest = max(wide, tall)
    response = httpx.get(
        BASEMAP_URL,
        params={
            "map": BASEMAP_MAP,
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": BASEMAP_LAYER,
            "STYLES": "",
            "SRS": "EPSG:4326",
            "BBOX": ",".join(
                f"{bound:.4f}" for bound in (box.west, box.south, box.east, box.north)
            ),
            "WIDTH": max(1, round(pixels * wide / longest)),
            "HEIGHT": max(1, round(pixels * tall / longest)),
            "FORMAT": "image/png",
        },
        timeout=BASEMAP_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()
    if not response.headers.get("content-type", "").startswith("image/"):
        raise ValueError(response.text.strip()[:200])
    return response.content
