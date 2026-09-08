"""How much memory the builds running at once may hold between them."""

from __future__ import annotations

import os
import threading
from collections import deque
from pathlib import Path

# Where a platform run is told, in bytes, the memory the box it runs on was given.
MEMORY_ENV = "PIPELINE_MEMORY_BYTES"

# Where a container writes the memory it is held to, by cgroup version.
CGROUP_LIMITS = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)


class Budget:
    """The memory a run hands out to its builds, in the order they ask for it.

    Attributes:
        total: How many bytes every build running at once may hold together,
            which is also the most any single one of them is given.
    """

    def __init__(self, total: int) -> None:
        """Open a budget over the memory a run was left.

        Args:
            total: How many bytes the builds may hold between them.
        """
        self.total = max(1, total)
        self._free = self.total
        self._queue: deque[object] = deque()
        self._changed = threading.Condition()

    def acquire(self, wanted: int) -> int:
        """Wait until one build's share is free, and take it.

        Args:
            wanted: How many bytes that build holds, which is cut to the whole
                budget where it asks for more than a run ever has.

        Returns:
            taken: How many bytes were taken, which is what has to be given back.
        """
        taken = min(max(wanted, 1), self.total)
        ticket = object()
        with self._changed:
            self._queue.append(ticket)
            while self._queue[0] is not ticket or self._free < taken:
                self._changed.wait()
            self._queue.popleft()
            self._free -= taken
            self._changed.notify_all()
        return taken

    def release(self, taken: int) -> None:
        """Give one build's share back to whoever is waiting for it.

        Args:
            taken: How many bytes that build held, as `acquire` handed them out.
        """
        with self._changed:
            self._free = min(self.total, self._free + taken)
            self._changed.notify_all()


def memory_bytes() -> int:
    """Return how many bytes of memory the box this run was given holds.

    Returns:
        held: The memory the platform named for the box, the limit a container
            is held to where one is written, and the machine's own free memory
            where neither says.
    """
    told = os.environ.get(MEMORY_ENV, "")
    if told.isdigit():
        return int(told)
    machine = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    for path in CGROUP_LIMITS:
        try:
            held = int(path.read_text().split()[0])
        except (OSError, ValueError):
            continue
        # An unlimited cgroup writes a sentinel larger than the machine itself
        if held <= machine:
            return held
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
