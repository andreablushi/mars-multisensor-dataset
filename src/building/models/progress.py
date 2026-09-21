"""Where every product of a running build is, so a stall is visible while it runs."""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field

# What a product is doing between being planned and being finished, in order.
QUEUED = "queued"
FETCHING = "fetching"
HOLDING = "holding"
BUILDING = "building"
STAGES = (QUEUED, FETCHING, HOLDING, BUILDING)

# How many of the products a build is waiting on it names, the longest held first.
NAMED = 3


@dataclass(slots=True)
class Progress:
    """How far a build has got, and what each product still in it is doing.

    Attributes:
        total: How many products the build has to get through.
        finished: How many are done, whether they were built or failed.
        moved_at: When a product last moved, as a monotonic reading.
    """

    total: int
    finished: int = 0
    moved_at: float = field(default_factory=time.monotonic)
    _held: dict[object, tuple[str, str, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def entered(self, label: str, stage: str) -> object:
        """Record that one product has reached the first stage of the build.

        Args:
            label: What the product is called, which is what a stall names.
            stage: The stage it has reached.

        Returns:
            ticket: What the product is tracked by, since two jobs of one
                archive's record are named alike.
        """
        ticket = object()
        with self._lock:
            self.moved_at = time.monotonic()
            self._held[ticket] = (label, stage, self.moved_at)
        return ticket

    def moved(self, ticket: object, onto: str) -> object:
        """Record that one product has moved on to the next stage.

        Args:
            ticket: What the product is tracked by, as `entered` handed it back.
            onto: The stage it has reached.

        Returns:
            ticket: That same ticket, so a caller holds one thing throughout.
        """
        with self._lock:
            self.moved_at = time.monotonic()
            label, _, _ = self._held[ticket]
            self._held[ticket] = (label, onto, self.moved_at)
        return ticket

    def left(self, ticket: object, *, finished: bool = False) -> None:
        """Record that one product has left the stage it was at.

        Args:
            ticket: What the product is tracked by, as `entered` handed it back.
            finished: Whether it left the build altogether rather than moving on.
        """
        with self._lock:
            self._held.pop(ticket, None)
            self.finished += finished
            self.moved_at = time.monotonic()

    @property
    def standing(self) -> str:
        """Return one line saying what the build is doing right now.

        Returns:
            written: How many products are at each stage, how many are done, how
                long since any moved, and the products held longest at one stage,
                which is what a stalled build is waiting on.
        """
        with self._lock:
            now = time.monotonic()
            held = list(self._held.values())
            done, still = self.finished, now - self.moved_at
        at = Counter(stage for _, stage, _ in held)
        counted = ", ".join(f"{at[stage]} {stage}" for stage in STAGES)
        line = f"{done}/{self.total} done; {counted}; last moved {still:.0f}s ago"
        if not held:
            return line
        # Named rather than counted, since a count never says what is holding it up
        waiting = sorted(held, key=lambda one: one[2])[:NAMED]
        named = ", ".join(
            f"{label} {stage} {now - since:.0f}s" for label, stage, since in waiting
        )
        return f"{line}; waiting on {named}"
