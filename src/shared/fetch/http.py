"""How a request reaches a server that answers slowly, or does not answer at first."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from shared.disk.files import atomic_path
from shared.fetch.throttle import Throttle

# A box spanning a fifth of the planet takes ODE forty seconds to answer at any
# page size, the cost being the query and not the payload
REQUEST_TIMEOUT = 180.0
MAX_RETRIES = 20
BACKOFF_BASE = 0.5
# Ceiling on one backoff sleep, so many retries stay minutes rather than days
BACKOFF_MAX = 30.0
RETRYABLE_STATUS = frozenset({403, 429, 500, 502, 503, 504})
# Which of those mean the caller is asking too often, and so hold every thread
CROWDED_STATUS = frozenset({403, 429})
# Fewer tries for a transfer than a query, one running for minutes not seconds
STREAM_RETRIES = 5

# How long one query may be asked for in all, an attempt count bounding nothing
QUERY_DEADLINE = 900.0

# How long one transfer may run in all, so a trickling server is given up on
STREAM_DEADLINE = 3600.0

# The pause every request waits out, which one archive's refusal lengthens.
ARCHIVE = Throttle()


class FetchError(RuntimeError):
    """Raised when a server refuses a request, or keeps failing to answer one."""


def slept(attempt: int, backoff: float) -> None:
    """Wait out one server's refusal, longer each time and never in step.

    Args:
        attempt: Which retry is about to be made, counting the first as one.
        backoff: The base delay, in seconds.
    """
    time.sleep(
        min(backoff * 2 ** (attempt - 1), BACKOFF_MAX) + random.uniform(0.0, backoff)
    )


def fetched_json(
    url: str,
    params: dict[str, str],
    *,
    accepted: Callable[[Any], Any | None],
    client: httpx.Client | None = None,
    timeout: float = REQUEST_TIMEOUT,
    retries: int = MAX_RETRIES,
    backoff: float = BACKOFF_BASE,
    deadline: float = QUERY_DEADLINE,
) -> Any:
    """Read one JSON reply, asking again until the server answers a usable one.

    Args:
        url: Where to ask.
        params: What to ask for.
        accepted: What reads the wanted part out of one reply, and hands back
            None for a reply worth asking again for.
        client: A client whose connections to reuse, or None to ask on its own.
        timeout: How long to wait on one attempt.
        retries: How many times to ask again after the first attempt.
        backoff: The base delay between attempts, in seconds.
        deadline: How long to keep asking for in all, in seconds.

    Returns:
        found: What `accepted` read out of the first usable reply.

    Raises:
        FetchError: When the server refuses the request, when no attempt left a
            reply `accepted` could read, or when the deadline passed first.
    """
    asking = client or httpx
    give_up_at = time.monotonic() + deadline
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            slept(attempt, backoff)
        if time.monotonic() >= give_up_at:
            break
        ARCHIVE.wait()
        try:
            reply = asking.get(url, params=params, timeout=timeout)
        except httpx.HTTPError as error:
            last = error
            continue
        if reply.status_code in RETRYABLE_STATUS:
            # One refusal slows every thread, so a run stops asking to be blocked.
            if reply.status_code in CROWDED_STATUS:
                ARCHIVE.refused()
            last = FetchError(f"HTTP {reply.status_code}")
            continue
        ARCHIVE.answered()
        if reply.status_code >= 400:
            raise FetchError(f"{url} refused the request: HTTP {reply.status_code}")
        try:
            payload = reply.json()
        except ValueError as error:
            last = error
            continue
        found = accepted(payload)
        if found is not None:
            return found
        last = FetchError("the reply held nothing to read")
    raise FetchError(f"gave up after {deadline:.0f}s or {retries} retries: {last}")


def streamed(
    url: str,
    path: Path,
    timeout: float,
    *,
    client: httpx.Client | None = None,
    retries: int = STREAM_RETRIES,
    backoff: float = BACKOFF_BASE,
    deadline: float = STREAM_DEADLINE,
) -> None:
    """Stream one file to disk, asking again while the server keeps failing.

    Args:
        url: Where to read it from.
        path: Where it belongs once it is whole.
        timeout: How long to wait on one transfer, between one chunk and the next.
        client: A client whose connections to reuse, or None to open one for this
            transfer alone.
        retries: How many times to ask again after the first attempt.
        backoff: The base delay between attempts, in seconds.
        deadline: How long the whole transfer may run for, in seconds.

    Raises:
        FetchError: When the server refuses the file, when every attempt fails,
            or when the deadline passed first.
    """
    reading = client.stream if client else httpx.stream
    give_up_at = time.monotonic() + deadline
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            slept(attempt, backoff)
        if time.monotonic() >= give_up_at:
            break
        ARCHIVE.wait()
        try:
            with reading("GET", url, timeout=timeout) as reply:
                if reply.status_code in RETRYABLE_STATUS:
                    if reply.status_code in CROWDED_STATUS:
                        ARCHIVE.refused()
                    last = FetchError(f"HTTP {reply.status_code}")
                    continue
                if reply.status_code >= 400:
                    raise FetchError(
                        f"{url} refused the request: HTTP {reply.status_code}"
                    )
                ARCHIVE.answered()
                # Nothing is left behind when a transfer fails part way through.
                with atomic_path(path) as tmp, tmp.open("wb") as handle:
                    for chunk in reply.iter_bytes():
                        # A timeout bounds one chunk, and this the whole transfer
                        if time.monotonic() >= give_up_at:
                            raise FetchError(
                                f"{url} was still sending after {deadline:.0f}s"
                            )
                        handle.write(chunk)
                return
        except httpx.HTTPError as error:
            last = error
    raise FetchError(f"gave up after {deadline:.0f}s or {retries} retries: {last}")
