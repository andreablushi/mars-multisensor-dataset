"""What every notebook's panels share: the basemap, the tile outline, the wording."""

from __future__ import annotations

import ipywidgets as widgets

GREY = "#8a8a8a"
STATISTIC_VALUE = ("Statistic", "Value")

BASEMAP_URL = "https://planetarymaps.usgs.gov/cgi-bin/mapserv"
BASEMAP_MAP = "/maps/mars/mars_simp_cyl.map"
BASEMAP_LAYER = "THEMIS"
BASEMAP_PIXELS = 900
BASEMAP_TIMEOUT = 30.0
BASEMAP_FAILED = "The basemap could not be fetched: {reason}"

NO_BOX = BASEMAP_FAILED.format(
    reason="this tile has no lon/lat box to crop the mosaic to"
)

MARS_PIXELS = 2400
MARS_FIGURE_SIZE = (14.0, 7.6)
GRATICULE = "#ffffff"

MIN_SPAN_DEG = 0.5
RING_SAMPLES = 17

TILE_EDGE = "#ffffff"
TILE_WIDTH = 1.4

PICKER = widgets.Layout(width="360px")
STEP = widgets.Layout(width="40px")

NOTHING = "none"
UNCOUNTED = "not counted"

STEPS = ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"))

LANDED = (
    "Instrument",
    "Mean observations selected",
    "Mean pixels landed per observation",
    "Pixels asked",
    "Mean coverage inside a tile",
)
