"""The pause every request waits out once an archive starts refusing them."""

from __future__ import annotations

import random
import threading
import time

# The first pause an archive's refusal buys, doubled by each refusal after it.
FIRST_PAUSE = 1.0

# The longest an archive may hold every request back, however often it refuses.
LONGEST_PAUSE = 60.0

# What each answered request takes off the pause, so it is given back gradually.
EASING = 0.9


class Throttle:
    """How hard every request to one archive is being held back.

    An archive refuses a caller that asks too often, and a run whose downloads
    each back off alone answers that by asking just as often from the threads
    that were not refused. The pause is held here instead, so one refusal slows
    the whole run and an archive that starts answering again is asked faster.
    """

    def __init__(self) -> None:
        """Open a throttle that holds nothing back until something refuses.

        Returns:
            None.
        """
        self._lock = threading.Lock()
        self._pause = 0.0
        self._until = 0.0

    def wait(self) -> None:
        """Wait out whatever pause the last refusal bought.

        Returns:
            None.
        """
        with self._lock:
            until, pause = self._until, self._pause
        left = until - time.monotonic()
        if left > 0:
            # Every waiter is held to the one moment, so they leave it spread out
            # rather than arriving together and being refused together.
            time.sleep(left + random.uniform(0.0, pause))

    def refused(self) -> None:
        """Hold every request back longer, one archive having refused this one.

        Returns:
            None.
        """
        with self._lock:
            self._pause = min(LONGEST_PAUSE, max(FIRST_PAUSE, self._pause * 2))
            self._until = time.monotonic() + self._pause

    def answered(self) -> None:
        """Give a little of the pause back, the archive having answered.

        Returns:
            None.
        """
        with self._lock:
            self._pause *= EASING
            if self._pause < FIRST_PAUSE:
                self._pause = 0.0
