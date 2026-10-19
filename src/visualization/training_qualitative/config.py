"""How the training qualitative notebook picks a tile and draws what landed on it."""

from __future__ import annotations

import ipywidgets as widgets

from building.configs import crism

COORDINATE = widgets.Layout(width="200px")
NO_TRACK = "The selection holds nothing to measure on this tile."

TRACE_WIDTH = 1.2

SURVEY = dict(color="#1a1a1a", linestyle=(0, (6, 3)), linewidth=0.8)

TITLE_BAND = 0.42

NOTHING_LANDED = "nothing on this tile"

STRETCH = (2, 98)
CLIP = 2.0
STD_FLOOR = 1e-6

ELEVATION_CMAP = "turbo"

DRAWN_PIXELS = 1000

ROW_HEIGHT = 4.5
CRISM_ROW_HEIGHT = 5.5

MOSAIC = "mosaic"

RAW_COUNTS = (0.0, 4095.0)

DOWNLOADED = "Downloaded sample"

BUILT_STAGES = (
    "Preprocessed sample",
    "Model input",
    "Min-max input",
    "Global min-max input",
)

STRETCHED = (DOWNLOADED, BUILT_STAGES[0])

RANGES = (
    "Array",
    "Cropped min (dB)",
    "Cropped max (dB)",
    "Preprocessed min (dB)",
    "Preprocessed max (dB)",
)

VIEWS = {
    "True colour R600 G530 B440": (crism.Detector.VISIBLE, (600.0, 530.0, 440.0)),
    "False colour R2529 G1506 B1080": (
        crism.Detector.INFRARED,
        (2529.0, 1506.0, 1080.0),
    ),
}
