"""What the filter left of every tile the selection searched, read as one."""

from __future__ import annotations

from dataclasses import dataclass

from analysis.stats.models.spread import Spread


@dataclass(frozen=True, slots=True)
class Aggregate:
    """What a run of tiles holds between them.

    Attributes:
        searched: How many tiles the search ran over.
        kept: How many of them earned a window worth keeping.
        days: How long the windows last, over the kept tiles.
        reached: The share of a tile each instrument reaches, over the kept.
        pixels_per_look: The pixels one observation of each instrument landed on
            a tile, over the kept tiles it took any of.
        pixel_km2: The ground one pixel of each instrument covers, over every
            tile searched, its size being the same on any of them.
    """

    searched: int
    kept: int
    days: Spread
    reached: dict[str, Spread]
    pixels_per_look: dict[str, Spread]
    pixel_km2: dict[str, Spread]


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """What the filter left of every tile searched.

    Attributes:
        held: Every tile searched, read as one.
        offered: How many observations each instrument landed on a tile searched.
        overlap: The share of a tile every instrument reaches at once, over the kept.
        iids: The instruments reported on, in the order they are drawn.
    """

    held: Aggregate
    offered: dict[str, Spread]
    overlap: Spread
    iids: list[str]
