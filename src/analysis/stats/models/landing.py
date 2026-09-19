"""What every observation offered to a tile landed on it, and the bar it faced."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Landed:
    """What one instrument set landed on a tile, look by look.

    Attributes:
        label: The set's short readable name.
        iid: The instrument it belongs to, which is what the filter names.
        counts: The pixels each of its observations landed on the tile, smallest
            first, the ones it turned away counted alongside the ones it kept.
        bar: The pixels the filter asks of it before a look counts as one.
    """

    label: str
    iid: str
    counts: list[float]
    bar: float
