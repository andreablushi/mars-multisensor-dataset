"""Where every product of a running build is, so a stall is visible while it runs."""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from building.models.job import Job

# How many of the products a build is waiting on it names, the longest held first.
NAMED = 3


class Stage(StrEnum):
    """What a product is doing between being planned and being finished, in order."""

    FETCHING = "fetching"
    PLACING = "placing"
    WAITING = "waiting"
    BUILDING = "building"


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
    _held: dict[Job, tuple[Stage, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def moved(self, job: Job, onto: Stage) -> None:
        """Record that one product has reached a stage, fetching being its first.

        Args:
            job: The product that moved.
            onto: The stage it has reached.
        """
        with self._lock:
            self.moved_at = time.monotonic()
            self._held[job] = (onto, self.moved_at)

    def finish(self, job: Job) -> None:
        """Record that one product has left the build, built or failed.

        Args:
            job: The product that left.
        """
        with self._lock:
            self._held.pop(job, None)
            self.finished += 1
            self.moved_at = time.monotonic()

    @property
    def standing(self) -> str:
        """Return one line saying what the build is doing right now.

        Returns:
            written: The products per stage, done, idle time and longest held.
        """
        with self._lock:
            now = time.monotonic()
            held = [(job.label, *at) for job, at in self._held.items()]
            done, still = self.finished, now - self.moved_at
        at = Counter(stage for _, stage, _ in held)
        counted = ", ".join(f"{at[stage]} {stage}" for stage in Stage)
        line = f"{done}/{self.total} done; {counted}; last moved {still:.0f}s ago"
        if not held:
            return line
        # Named rather than counted, since a count never says what is holding it up
        waiting = sorted(held, key=lambda one: one[2])[:NAMED]
        named = ", ".join(
            f"{label} {stage} {now - since:.0f}s" for label, stage, since in waiting
        )
        return f"{line}; waiting on {named}"
