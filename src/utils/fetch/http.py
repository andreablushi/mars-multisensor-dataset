"""How a request reaches a server that answers slowly, or does not answer at first."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from utils.disk.files import atomic_path
from utils.fetch.throttle import Throttle

REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 20
BACKOFF_BASE = 0.5
# Ceiling on one backoff sleep, so many retries stay minutes rather than days
BACKOFF_MAX = 30.0
RETRYABLE_STATUS = frozenset({403, 429, 500, 502, 503, 504})
# Which of those mean the caller is asking too often, as against a server that is
# merely broken or has nothing to give. Only these hold every other thread back:
# an archive answering for a file it does not have with a 500 would otherwise
# brake a whole run for asking it a question it invited.
CROWDED_STATUS = frozenset({403, 429})
# Fewer tries for a transfer than for a query, since one runs for minutes and
# a job retrying every one of them would hang for hours
STREAM_RETRIES = 5

# How long one query may be asked for in all, retries and their waits included.
# An attempt count alone bounds nothing: a server that answers slowly and then
# refuses leaves a thread here for the sum of its timeouts, which is an hour.
QUERY_DEADLINE = 420.0

# How long one transfer may run in all, the retries and the reading included, so
# a server that keeps a connection open while trickling is given up on.
STREAM_DEADLINE = 1800.0

# The pause every request waits out, which one archive's refusal lengthens.
ARCHIVE = Throttle()


class FetchError(RuntimeError):
    """Raised when a server refuses a request, or keeps failing to answer one."""


def slept(attempt: int, backoff: float) -> None:
    """Wait out one server's refusal, longer each time and never in step.

    Args:
        attempt: Which retry is about to be made, counting the first as one.
        backoff: The base delay, in seconds.

    Returns:
        None.
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
        What `accepted` read out of the first usable reply.

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
    retries: int = STREAM_RETRIES,
    backoff: float = BACKOFF_BASE,
    deadline: float = STREAM_DEADLINE,
) -> None:
    """Stream one file to disk, asking again while the server keeps failing.

    Args:
        url: Where to read it from.
        path: Where it belongs once it is whole.
        timeout: How long to wait on one transfer, between one chunk and the next.
        retries: How many times to ask again after the first attempt.
        backoff: The base delay between attempts, in seconds.
        deadline: How long the whole transfer may run for, in seconds.

    Returns:
        None.

    Raises:
        FetchError: When the server refuses the file, when every attempt fails,
            or when the deadline passed first.
    """
    give_up_at = time.monotonic() + deadline
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            slept(attempt, backoff)
        if time.monotonic() >= give_up_at:
            break
        ARCHIVE.wait()
        try:
            with httpx.stream("GET", url, timeout=timeout) as reply:
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
                        # A timeout bounds one chunk, and this the whole transfer,
                        # so a server that trickles is given up on rather than held.
                        if time.monotonic() >= give_up_at:
                            raise FetchError(
                                f"{url} was still sending after {deadline:.0f}s"
                            )
                        handle.write(chunk)
                return
        except httpx.HTTPError as error:
            last = error
    raise FetchError(f"gave up after {deadline:.0f}s or {retries} retries: {last}")
