"""How a measurement reads once it is put into words."""

from __future__ import annotations

from collections.abc import Callable

from common.analysis.stats.models.spread import Spread
from common.analysis.visualization.common import quantities

NOTHING = "none"
UNCOUNTED = "not counted"
SOUNDER = "SHARAD"


def counted(number: float, noun: str) -> str:
    """Write how many of something there are, the noun made plural to match."""
    return f"{number:,.0f} {noun}" + ("" if number == 1 else "s")


def spread(measured: Spread, written: Callable[[float], str]) -> str:
    """Write a measurement read off many tiles, and how far they sit from it."""
    if not measured.counted:
        return NOTHING
    middle = written(measured.mean)
    if measured.agreed:
        return middle
    return f"{middle} ± {written(measured.deviation)}"


def ground(km2: float, of_km2: float) -> str:
    """Write an amount of ground and what share of something it is."""
    if not km2:
        return NOTHING
    return f"{quantities.area(km2)}, {km2 / of_km2:.0%}"


def pixels(counted: float | None) -> str:
    """Write a pixel count, or that it was never measured."""
    if counted is None:
        return UNCOUNTED
    return f"{quantities.compact(counted)} px"
