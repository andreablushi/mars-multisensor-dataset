"""How the noise of a SHARAD crop is told apart from its echoes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Gate:
    """The settings a crop's noise is measured and gated with.

    Attributes:
        looks: How many traces each delay row is averaged over.
        false_alarm: The share of pure noise left over the threshold.
        guard_rows: The rows kept between an echo and the noise beside it.
        echo_margin_db: How far a row's mean power stands over the noise above the
            surface to hold an echo.
    """

    looks: int
    false_alarm: float
    guard_rows: int
    echo_margin_db: float
