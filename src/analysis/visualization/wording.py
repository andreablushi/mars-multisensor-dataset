"""How a measurement reads once it is put into words."""

from __future__ import annotations

from collections.abc import Callable

from analysis.stats.models import Spread

NOTHING = "none"
UNCOUNTED = "not counted"
SOUNDER = "SHARAD"

_STEPS = ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"))


def compact(value: float) -> str:
    """Write a count short enough to read at a glance.

    Args:
        value: The count.

    Returns:
        written: The count itself when small, and otherwise in thousands and up.
    """
    for limit, suffix in _STEPS:
        if value >= limit:
            return f"{value / limit:,.2f} {suffix}"
    return f"{value:,.0f}"


def area(km2: float) -> str:
    """Write a ground area, to a hundredth below ten km2 and whole above."""
    return f"{km2:,.0f} km2" if km2 >= 10.0 else f"{km2:,.2f} km2"


def duration(days: float) -> str:
    """Write a length of time in the units it reads well in.

    Args:
        days: The length in days.

    Returns:
        written: The length as a phrase, such as "18 hours" or "47 days".
    """
    if days < 1.0:
        return f"{days * 24.0:.0f} hours"
    if days < 10.0:
        return f"{days:.1f} days"
    return f"{days:,.0f} days"


def counted(number: float, noun: str) -> str:
    """Write how many of something there are, the noun made plural to match."""
    return f"{number:,.0f} {noun}" + ("" if number == 1 else "s")


def pixels(count: float | None) -> str:
    """Write a pixel count, or that it was never measured.

    Args:
        count: The pixel count, or None where it was never measured.

    Returns:
        written: The count in pixels, or that it was not counted.
    """
    if count is None:
        return UNCOUNTED
    return f"{compact(count)} px"


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
