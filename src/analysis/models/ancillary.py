"""The signal phase distortion of each SHARAD look, read off its geometry table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Distortion:
    """The least signal phase distortion of one SHARAD look's rows over one tile.

    Attributes:
        group: The tile group the look was listed in.
        tile: The tile its rows fall on, such as "b123_c0456".
        pdsid: The PDS product identifier.
        night: The least of its rows on the night side, or None when none were.
        overall: The least of every row.
    """

    group: str
    tile: str
    pdsid: str
    night: float | None
    overall: float
