"""What every statistic is handed back as, for one tile and for all of them."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from analysis.selector.models.selection import SelectedTile
from analysis.selector.models.track import Track


@dataclass(frozen=True, slots=True)
class Spread:
    """The same number read off many tiles.

    Attributes:
        mean: Their average.
        middle: Their median, which a handful of wide tiles cannot pull.
        low: The least of them.
        high: The most of them.
        counted: How many there were.
    """

    mean: float
    middle: float
    low: float
    high: float
    counted: int

    @classmethod
    def over(cls, values: Sequence[float]) -> Spread:
        """Read one measurement off every tile that took it.

        Args:
            values: The measurement, one per tile, in any order.

        Returns:
            spread: The spread, empty at nought where no tile took it.
        """
        if not values:
            return cls(0.0, 0.0, 0.0, 0.0, 0)
        return cls(
            mean=statistics.fmean(values),
            middle=statistics.median(values),
            low=min(values),
            high=max(values),
            counted=len(values),
        )

    @property
    def agreed(self) -> bool:
        """Report whether the tiles all read the same, so the average says it all."""
        return self.low == self.high


@dataclass(frozen=True, slots=True)
class InstrumentStats:
    """What one instrument holds of the whole measured dataset.

    Attributes:
        iid: The instrument, such as CTX.
        tiles: How many tiles it reached.
        observations: How many observations it took, once per tile reached.
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
        instruments: What each instrument holds, in the order they are drawn.
    """

    tiles: int
    tile_km: float
    measured: int
    tile_km2: Spread
    instruments: list[InstrumentStats]


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """What the filter left of every tile searched.

    Attributes:
        searched: How many tiles the search ran over.
        kept: How many of them earned a window worth keeping.
        days: How long the windows last, over the kept tiles.
        reached: The share of a tile each instrument reaches, over the kept.
        landed_km2_per_look: The ground one observation lands on a tile, per instrument.
        selected: How many observations of each instrument a kept tile keeps.
        downloads: How many distinct products of each instrument the kept tiles keep.
        overlap: The share of a tile every instrument reaches at once, over the kept.
        iids: The instruments reported on, in the order they are drawn.
    """

    searched: int
    kept: int
    days: Spread
    reached: dict[str, Spread]
    landed_km2_per_look: dict[str, Spread]
    selected: dict[str, Spread]
    downloads: dict[str, int]
    overlap: Spread
    iids: list[str]


@dataclass(frozen=True, slots=True)
class TileTrack:
    """One tile's track, the window it earned, and where the observations it keeps sit.

    Attributes:
        track: Its admissible observations on one time axis.
        window: The window the selection gave it, or refused it.
        taken: Where the observations it keeps sit on that axis, oldest first.
    """

    track: Track
    window: SelectedTile
    taken: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class InstrumentReach:
    """What one instrument left on one tile inside its window.

    Attributes:
        km2: The ground it reaches, counting a cell once however often it was revisited.
        landed_km2: The ground its observations landed there, revisits counted again.
        observations_taken: How many of its observations the window keeps.
    """

    km2: float
    landed_km2: float
    observations_taken: int

    @property
    def landed_km2_per_look(self) -> float:
        """Return the ground one of its observations lands on the tile.

        Returns:
            km2: The mean over the window's observations.
        """
        return self.landed_km2 / self.observations_taken


@dataclass(frozen=True, slots=True)
class TileStats:
    """One tile, and what the observations it keeps left on it.

    Attributes:
        window: The window the selection gave it, with its name, box and span.
        iids: The instruments it holds, in the order they are drawn.
        reached: What each instrument left on it, by instrument.
        overlaps: The ground each set of instruments reaches, most ground first.
    """

    window: SelectedTile
    iids: list[str]
    reached: dict[str, InstrumentReach]
    overlaps: dict[tuple[str, ...], float]


@dataclass(frozen=True, slots=True)
class Landing:
    """What one instrument set landed on a tile, observation by observation.

    Attributes:
        label: The set's short readable name.
        iid: The instrument it belongs to, which is what the filter names.
        landed_km2: The ground each observation landed on the tile, smallest first.
        bar_km2: The ground the filter asks of it before a look counts as one.
    """

    label: str
    iid: str
    landed_km2: list[float]
    bar_km2: float


@dataclass(frozen=True, slots=True)
class Timeline:
    """What one instrument set observed of the ground on show.

    Attributes:
        label: The set's short readable name.
        iid: The instrument it belongs to, which is what the filter names.
        times: When each of its observations started, oldest first.
        shares: How much of the ground each of them covered on its own.
        running: How much of the ground it had reached by then, revisits counted once.
        covered: The share it ends on.
        first: The earliest moment it is drawn from.
        last: The latest moment it is drawn to.
        reason: Why it holds nothing to draw, and empty when it observed.
    """

    label: str
    iid: str
    times: list[datetime]
    shares: list[float]
    running: list[float]
    covered: float
    first: datetime
    last: datetime
    reason: str

    @property
    def observed(self) -> bool:
        """Report whether the set holds any observation of the ground on show."""
        return bool(self.times)
