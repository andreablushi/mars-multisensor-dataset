"""What the lava flow annotation calls a verdict, heads its table, and says if empty."""

from __future__ import annotations

from mars_multisensor_dataset.analysis.ground_truth.lavaflow.models import (
    NONE,
    REFLECTOR,
    UNSURE,
)

BUTTON_STYLES = {REFLECTOR: "success", NONE: "danger", UNSURE: "warning"}
NOT_REVIEWED = "not reviewed"
REVIEW = (
    "Region",
    "Source",
    "Crops",
    "Of named tracks",
    *(verdict.capitalize() for verdict in BUTTON_STYLES),
    "Not reviewed",
)
NOTHING_LISTED = "No lava flow crop has been listed."
NOT_BUILT = "This crop is not on disk. Build or download the lava flow build first."
FIGURE_SIZE = (14.0, 7.0)
KINDS = ("Radargram, gated (dB)", "Cluttergram (dB)")
