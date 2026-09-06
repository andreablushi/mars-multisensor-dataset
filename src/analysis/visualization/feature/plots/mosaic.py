"""The mosaic under a feature: fetching one crop of it, and drawing it."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable
from functools import lru_cache
from html import escape

import httpx
import ipywidgets as widgets
from matplotlib import image as reading
from matplotlib.axes import Axes

from analysis.visualization.common import panels
from analysis.visualization.feature.models.placing import Box
from utils.geometry import geodesy

BASEMAP_URL = "https://planetarymaps.usgs.gov/cgi-bin/mapserv"
BASEMAP_MAP = "/maps/mars/mars_simp_cyl.map"
BASEMAP_LAYER = "THEMIS"
BASEMAP_PIXELS = 900
BASEMAP_TIMEOUT = 30.0
BASEMAP_FAILED = "The basemap could not be fetched: {reason}"
BASEMAP_LOADING = "Fetching the basemap..."
BASEMAP_CACHE = 32

PLACEHOLDER = "320px"

NO_BOX = "this feature has no lon/lat box to crop the mosaic to"


def fetched(box: Box, draw: Callable[[bytes], widgets.Widget]) -> widgets.Box:
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
            image = crop(box)
        except Exception as exc:
            space.children = (panels.unavailable(BASEMAP_FAILED.format(reason=exc)),)
            return
        space.children = (draw(image),)

    threading.Thread(target=fill, daemon=True).start()
    return space


def draw(axis: Axes, box: Box, image: bytes) -> None:
    """Draw one mosaic crop onto an axis in lon and lat."""
    axis.imshow(
        reading.imread(io.BytesIO(image), format="png"),
        extent=box.extent,
        origin="upper",
        cmap="gray",
    )
    axis.set_aspect(1.0 / geodesy.longitude_stretch(box.centre_lat))
    axis.set_xlim(box.west, box.east)
    axis.set_ylim(box.south, box.north)
    # A footprint reaching well past the crop is cut to it rather than framed
    axis.autoscale(False)


@lru_cache(maxsize=BASEMAP_CACHE)
def crop(box: Box) -> bytes:
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
            "WIDTH": max(1, round(BASEMAP_PIXELS * wide / longest)),
            "HEIGHT": max(1, round(BASEMAP_PIXELS * tall / longest)),
            "FORMAT": "image/png",
        },
        timeout=BASEMAP_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()
    if not response.headers.get("content-type", "").startswith("image/"):
        raise ValueError(response.text.strip()[:200])
    return response.content
