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

VIEWS = {
    "True colour R600 G530 B440": (crism.Detector.VISIBLE, (600.0, 530.0, 440.0)),
    "False colour R2529 G1506 B1080": (
        crism.Detector.INFRARED,
        (2529.0, 1506.0, 1080.0),
    ),
}
