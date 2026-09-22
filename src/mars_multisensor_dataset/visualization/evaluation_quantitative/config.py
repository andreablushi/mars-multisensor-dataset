"""How the evaluation notebook heads its classes table and marks a drawn tile."""

from __future__ import annotations

CLASSES = (
    "Class",
    "Read from",
    "Tiles labelled",
    "Features labelled",
    "Tiles drawn",
    "Features drawn",
)
NOTHING_DRAWN = "No tile has been drawn into the evaluation set."
MARKER_SIZE = 18

TILE_MARKER = 1.5
CATEGORIES = (
    "Icy at both depths",
    "Icy at 1 to 5 m only",
    "Icy below 5 m only",
    "Icy at neither",
)
