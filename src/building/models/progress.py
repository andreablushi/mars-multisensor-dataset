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


@dataclass(slots=True)
class Progress:
    """How far a build has got, and what each product still in it is doing.

    A build waits for a turn on the network, on the archive that answers, on
    memory and on its cores in turn, and a run that stops moving looks the same
    from outside whichever of them it stopped on. Counting the products at each
    stage says which, and when one last moved says whether it is slow or
    stopped. Waiting for a turn is counted apart from being answered, so the
    line never reads as more downloads than a run is allowed to run.

    Attributes:
        total: How many products the build has to get through.
        finished: How many are done, whether they were built or failed.
        moved_at: When a product last moved, as a monotonic reading.
    """

    total: int
    finished: int = 0
    moved_at: float = field(default_factory=time.monotonic)
    _at: Counter[str] = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def entered(self, stage: str) -> str:
        """Record that one more product has reached a stage.

        Args:
            stage: The stage it has reached.

        Returns:
            That stage, so a caller can hold it as where the product now is.
        """
        with self._lock:
            self._at[stage] += 1
            self.moved_at = time.monotonic()
        return stage

    def left(self, stage: str, *, finished: bool = False) -> None:
        """Record that one product has left a stage.

        Args:
            stage: The stage it has left.
            finished: Whether it left the build altogether rather than moving on.

        Returns:
            None.
        """
        with self._lock:
            self._at[stage] -= 1
            self.finished += finished
            self.moved_at = time.monotonic()

    def moved(self, stage: str, onto: str) -> str:
        """Record that one product has moved from one stage to the next.

        Args:
            stage: The stage it has left.
            onto: The stage it has reached.

        Returns:
            The stage it has reached.
        """
        self.left(stage)
        return self.entered(onto)

    @property
    def standing(self) -> str:
        """Return one line saying what the build is doing right now.

        Returns:
            How many products are at each stage, how many are done, and how
            long it has been since any of them moved.
        """
        with self._lock:
            at = {stage: self._at[stage] for stage in STAGES}
            done, still = self.finished, time.monotonic() - self.moved_at
        counted = ", ".join(f"{count} {stage}" for stage, count in at.items())
        return f"{done}/{self.total} done; {counted}; last moved {still:.0f}s ago"
