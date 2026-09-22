"""What the annotation notebook calls a verdict, heads its table, and says if empty."""

from __future__ import annotations

VERDICT = {True: "accepted", False: "rejected", None: "not reviewed"}
VERDICTS = (
    "Class",
    "Tiles labelled",
    "Accepted",
    "Rejected",
    "Drawn now",
    "Drawn, not reviewed",
)
NOTHING_LABELLED = "No tile has been labelled."
