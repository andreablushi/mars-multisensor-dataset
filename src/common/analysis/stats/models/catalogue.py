"""What the measured dataset holds, before the filter is asked of it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from common.analysis.stats.models.spread import Spread


@dataclass(frozen=True, slots=True)
class InstrumentStats:
    """What one instrument holds of the whole measured dataset.

    Attributes:
        iid: The instrument, such as CTX.
        tiles: How many tiles it reached.
        observations: How many observations of them it took, a look counted once
            per tile it reached.
        first: When the earliest of its observations was taken.
        last: When the latest of them was taken.
    """

    iid: str
    tiles: int
    observations: int
    first: datetime
    last: datetime


@dataclass(frozen=True, slots=True)
class CatalogueStats:
    """What the measured dataset holds, whatever the filter would make of it.

    Attributes:
        tiles: How many tiles Mars is split into altogether.
        tile_km: The side every tile is sized to, in kilometres.
        measured: How many of them any instrument reached.
        tile_km2: How much ground a measured tile holds, tile by tile.
        instruments: What each instrument holds, most observations first.
    """

    tiles: int
    tile_km: float
    measured: int
    tile_km2: Spread
    instruments: list[InstrumentStats]
