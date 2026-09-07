"""How much memory the builds running at once may hold between them."""

from __future__ import annotations

import threading
from collections import deque


class Budget:
    """The memory a run hands out to its builds, in the order they ask for it.

    A build holds a whole product, and the largest of them is many times the
    size of the smallest, so what may run at once is settled by what each one
    holds rather than by a count. Asking in order keeps a large build from
    waiting for ever behind an unbroken run of small ones.

    Attributes:
        total: How many bytes every build running at once may hold together,
            which is also the most any single one of them is given.
    """

    def __init__(self, total: int) -> None:
        """Open a budget over the memory a run was left.

        Args:
            total: How many bytes the builds may hold between them.

        Returns:
            None.
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
            How many bytes were taken, which is what has to be given back.
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

        Returns:
            None.
        """
        with self._changed:
            self._free = min(self.total, self._free + taken)
            self._changed.notify_all()
