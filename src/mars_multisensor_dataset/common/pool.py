"""Pools whose queued work is dropped when the block they serve is interrupted."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Executor
from contextlib import contextmanager


@contextmanager
def cancellable_pool[Pool: Executor](pool: Pool) -> Iterator[Pool]:
    """Yield a pool that cancels its queued work when the block raises.

    Args:
        pool: The executor to hand out and shut down on leaving.

    Yields:
        pool: The same executor.
    """
    try:
        yield pool
    except BaseException:
        pool.shutdown(cancel_futures=True)
        raise
    pool.shutdown()
