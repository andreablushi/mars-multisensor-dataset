"""How a measurement reads once it is put into words."""

from __future__ import annotations

from collections.abc import Callable

from analysis.stats.models.spread import Spread
from analysis.visualization.common import quantities

NOTHING = "none"
UNCOUNTED = "not counted"
SOUNDER = "SHARAD"


def counted(number: float, noun: str) -> str:
    """Write how many of something there are, the noun made plural to match."""
    return f"{number:,.0f} {noun}" + ("" if number == 1 else "s")


def spread(
    measured: Spread,
    written: Callable[[float], str],
    bounds: Callable[[float], str] | None = None,
) -> str:
    """Write the mean of a measurement read off many tiles, then its least and most.

    Args:
        measured: The measurement read off every tile.
        written: How the mean is written.
        bounds: How the least and the most are written, or None to write them alike.

    Returns:
        written: The mean, then the least and the most unless every tile agreed.
    """
    if not measured.counted:
        return NOTHING
    mean = written(measured.mean)
    if measured.agreed:
        return mean
    low, high = map(bounds or written, (measured.low, measured.high))
    return f"{mean} ({low} to {high})"


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
