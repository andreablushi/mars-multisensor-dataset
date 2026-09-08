"""One instrument set's observations of a feature over time, ready to draw."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Series:
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
        """Report whether the set holds any observation of the ground on show.

        Returns:
            observed: True when it holds at least one.
        """
        return bool(self.times)
