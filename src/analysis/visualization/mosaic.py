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

from analysis.visualization import panels
from common.fetch.http import TLS_CONTEXT
from common.maths import geodesy
from common.maths.box import Crop
from common.maths.physics import RADIUS_M

BASEMAP_URL = "https://planetarymaps.usgs.gov/cgi-bin/mapserv"
BASEMAP_MAP = "/maps/mars/mars_simp_cyl.map"
BASEMAP_LAYER = "THEMIS"
BASEMAP_PIXELS = 900
BASEMAP_TIMEOUT = 30.0
BASEMAP_FAILED = "The basemap could not be fetched: {reason}"

NO_BOX = BASEMAP_FAILED.format(
    reason="this tile has no lon/lat box to crop the mosaic to"
)

MARS = Crop(-180.0, -90.0, 180.0, 90.0)
MARS_PIXELS = 2400
MARS_FIGURE_SIZE = (14.0, 7.6)

GLOBE = crs.Globe(semimajor_axis=RADIUS_M, semiminor_axis=RADIUS_M, ellipse=None)
LONLAT = crs.PlateCarree(globe=GLOBE)
ROBINSON = crs.Robinson(globe=GLOBE)
GRATICULE = "#ffffff"

MARS_LAID = dict(
    extent=MARS.extent,
    origin="upper",
    interpolation="nearest",
    transform=LONLAT,
    regrid_shape=(MARS_PIXELS, MARS_PIXELS // 2),
)


def fetched(
    box: Crop, draw: Callable[[bytes], widgets.Widget], pixels: int = BASEMAP_PIXELS
) -> widgets.Box:
    """Claim the space one crop goes in and fill it off the thread that fetches it.

    Args:
        box: The lon/lat box the crop covers.
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
                f" border-radius: 4px; color: {panels.GREY};"
                f" font-family: sans-serif; font-size: 12px;'>"
                f"Fetching the basemap...</div>"
            )
        ]
    )
    threading.Thread(
        target=lambda: _fill(space, box, draw, pixels), daemon=True
    ).start()
    return space


def _fill(
    space: widgets.Box, box: Crop, draw: Callable[[bytes], widgets.Widget], pixels: int
) -> None:
    """Fetch one crop and put the figure drawn from it in the claimed space.

    Args:
        space: The space claimed for the figure.
        box: The lon/lat box the crop covers.
        draw: What turns the fetched crop into the figure shown.
        pixels: How many pixels the crop's longer side is fetched at.
    """
    try:
        image = crop(box, pixels)
    except Exception as exc:
        space.children = (panels.unavailable(BASEMAP_FAILED.format(reason=exc)),)
        return
    space.children = (draw(image),)


def read_mosaic(image: bytes) -> np.ndarray:
    """Decode one mosaic crop as fetched.

    Args:
        image: The crop, as `crop` hands it back.

    Returns:
        pixels: Its pixels, rows from the north.
    """
    return imread(io.BytesIO(image), format="png")


def board(size: tuple[float, float], box: Crop, image: bytes) -> tuple[Figure, Axes]:
    """Open a figure with one mosaic crop drawn on it, labelled in lon and lat.

    Args:
        size: The figure's size in inches.
        box: The lon/lat box the crop covers.
        image: The crop, as `crop` hands it back.

    Returns:
        figure: The figure the crop is drawn on.
        axis: The crop itself, in lon and lat.
    """
    figure, axis = panels.board(size)
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
    return figure, axis


def mars_board(image: bytes, title: str) -> tuple[Figure, Axes]:
    """Open a figure with the mosaic of Mars drawn under its graticule.

    Args:
        image: The mosaic of the whole planet, as `crop` hands it back.
        title: What the map is titled.

    Returns:
        figure: The figure the map is drawn on.
        axis: The map itself, in lon and lat.
    """
    figure, axis = panels.board(MARS_FIGURE_SIZE, ROBINSON)
    axis.set_global()
    axis.imshow(read_mosaic(image), cmap="gray", **MARS_LAID)
    lines = axis.gridlines(
        LONLAT, draw_labels=True, color=GRATICULE, linewidth=0.4, alpha=0.5
    )
    lines.xlabel_style = lines.ylabel_style = {"size": 8}
    axis.set_title(title, fontsize=12, loc="left")
    return figure, axis


@lru_cache(maxsize=32)
def crop(box: Crop, pixels: int = BASEMAP_PIXELS) -> bytes:
    """Fetch the mosaic over one lon/lat box, held for the panels sharing it.

    Args:
        box: The lon/lat box to fetch.
        pixels: How many pixels the crop's longer side is fetched at.

    Returns:
        image: The crop, as the PNG the map server sent.

    Raises:
        ValueError: When the server answers with something other than an image.
    """
    tall = box.north - box.south
    wide = (box.east - box.west) * geodesy.longitude_stretch(box.centre_lat)
    longest = max(wide, tall)
    response = httpx.get(
        BASEMAP_URL,
        verify=TLS_CONTEXT,
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
