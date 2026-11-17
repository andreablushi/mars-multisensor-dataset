"""What the lava flow annotation calls a verdict, heads its table, and says if empty."""

from __future__ import annotations

from mars_multisensor_dataset.analysis.ground_truth.lavaflow.models import (
    NONE,
    REFLECTOR,
    UNSURE,
)

BUTTON_STYLES = {REFLECTOR: "success", NONE: "danger", UNSURE: "warning"}
NOT_REVIEWED = "not reviewed"
FROM_PAPER = " [from the paper]"
FROM_PAPER_GROUP = "from the paper"
NOT_NAMED_GROUP = "not named"
ALL_GROUP = "all"
REVIEW = (
    "Region",
    "Source",
    "Tracks of",
    "Tracks",
    "Tiles",
    "Crops",
    *(verdict.capitalize() for verdict in BUTTON_STYLES),
    "Not reviewed",
)
NOTHING_LISTED = "No lava flow crop has been listed."
NOT_BUILT = (
    "This crop is not on disk. Build or download the lava flow candidates first."
)
FIGURE_SIZE = (14.0, 7.0)
KINDS = ("Radargram, gated (dB)", "Cluttergram (dB)")
