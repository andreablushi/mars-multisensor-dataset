"""How a measured quantity is scaled, and written short enough to read."""

from __future__ import annotations

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
    """Write a ground area short enough to read at a glance.

    Args:
        km2: The area in square kilometres.

    Returns:
        written: The area, to a hundredth below ten square kilometres and whole above
            it.
    """
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
