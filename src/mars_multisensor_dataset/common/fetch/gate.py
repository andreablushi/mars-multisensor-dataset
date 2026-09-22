"""The gate a refusing server is asked through, probed by one lane at a time."""

from __future__ import annotations

import threading
import time

FIRST_PROBE = 5.0

LONGEST_PROBE = 15.0


class Gate:
    """Whether a server is taking requests, probed by one lane while it refuses."""

    def __init__(self) -> None:
        """Open a gate that lets every request through until one is refused."""
        self._changed = threading.Condition()
        self._closed = False
        self._interval = 0.0
        self._next = 0.0
        self._prober: int | None = None

    def wait(self, until: float) -> bool:
        """Wait until a request may go, as the probe while the server refuses.

        Args:
            until: When to stop waiting, on the monotonic clock.

        Returns:
            open: True when the request may go, False when the wait ran out.
        """
        with self._changed:
            while self._closed:
                now = time.monotonic()
                if now >= until:
                    return False
                if self._prober is None and now >= self._next:
                    self._prober = threading.get_ident()
                    return True
                # A lane wakes for its probe turn, or when the probe settles.
                wake = until if self._prober is not None else min(until, self._next)
                self._changed.wait(max(0.0, wake - now))
            return True

    def refused(self) -> None:
        """Close the gate, waiting longer only when the probe itself was refused."""
        with self._changed:
            if not self._closed:
                self._closed, self._interval = True, FIRST_PROBE
            elif self._prober == threading.get_ident():
                self._interval = min(LONGEST_PROBE, self._interval * 2)
            else:
                # A request sent before the gate closed tells nothing new.
                return
            self._prober = None
            self._next = time.monotonic() + self._interval
            self._changed.notify_all()

    def answered(self) -> None:
        """Open the gate, the server having answered, and wake every waiting lane."""
        with self._changed:
            if self._closed:
                self._closed, self._prober = False, None
                self._changed.notify_all()
