"""How a request reaches a server that answers slowly, or does not answer at first."""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from common.disk.files import atomic_path
from common.fetch.throttle import Throttle

# A wide box takes ODE forty seconds at any page size, the query being costly
REQUEST_TIMEOUT = 180.0
MAX_RETRIES = 20
BACKOFF_BASE = 0.5
# Ceiling on one backoff sleep, so many retries stay minutes rather than days
BACKOFF_MAX = 30.0
RETRYABLE_STATUS = frozenset({403, 429, 500, 502, 503, 504})
# Which of those mean the caller is asking too often, and so hold the host back
CROWDED_STATUS = frozenset({403, 429})
# Fewer tries for a transfer than a query, one running for minutes not seconds
STREAM_RETRIES = 5

CONNECT_TIMEOUT = 30.0

CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)

# How long one query may be asked for in all, an attempt count bounding nothing
QUERY_DEADLINE = 900.0

# How long one transfer may run in all, so a trickling server is given up on
STREAM_DEADLINE = 10800.0


@functools.cache
def throttle(host: str) -> Throttle:
    """Return the pause every request to one host waits out.

    Args:
        host: The host asked, whose refusals hold back its requests alone.

    Returns:
        throttle: The one throttle that host shares across threads.
    """
    return Throttle()


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
        accepted: What reads one reply, or hands back None to ask again.
        client: A client whose connections to reuse, or None to ask on its own.
        timeout: How long to wait on one attempt.
        retries: How many times to ask again after the first attempt.
        backoff: The base delay between attempts, in seconds.
        deadline: How long to keep asking for in all, in seconds.

    Returns:
        found: What `accepted` read out of the first usable reply.

    Raises:
        FetchError: When refused, when no reply was readable, or past the deadline.
    """
    asking = client or httpx
    archive = throttle(httpx.URL(url).host)
    give_up_at = time.monotonic() + deadline
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            slept(attempt, backoff)
        if time.monotonic() >= give_up_at:
            break
        archive.wait()
        try:
            reply = asking.get(
                url,
                params=params,
                timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT),
            )
        except httpx.HTTPError as error:
            if isinstance(error, CONNECT_ERRORS):
                archive.refused()
            last = error
            continue
        if reply.status_code in RETRYABLE_STATUS:
            # One refusal slows every thread on that host, so a run stops being blocked.
            if reply.status_code in CROWDED_STATUS:
                archive.refused()
            last = FetchError(f"HTTP {reply.status_code}")
            continue
        archive.answered()
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
    span: tuple[int, int] | None = None,
) -> None:
    """Stream one file to disk, asking again while the server keeps failing.

    Args:
        url: Where to read it from.
        path: Where it belongs once it is whole.
        timeout: How long to wait on one transfer, between one chunk and the next.
        client: A client whose connections to reuse, or None to open one.
        retries: How many times to ask again after the first attempt.
        backoff: The base delay between attempts, in seconds.
        deadline: How long the whole transfer may run for, in seconds.
        span: The first and past-the-last byte to keep, or None for the whole file.

    Raises:
        FetchError: When refused, when every attempt fails, or past the deadline.
    """
    reading = client.stream if client else httpx.stream
    headers = {"Range": f"bytes={span[0]}-{span[1] - 1}"} if span else None
    archive = throttle(httpx.URL(url).host)
    give_up_at = time.monotonic() + deadline
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            slept(attempt, backoff)
        if time.monotonic() >= give_up_at:
            break
        archive.wait()
        try:
            with reading(
                "GET",
                url,
                timeout=httpx.Timeout(timeout, connect=CONNECT_TIMEOUT),
                headers=headers,
            ) as reply:
                if reply.status_code in RETRYABLE_STATUS:
                    if reply.status_code in CROWDED_STATUS:
                        archive.refused()
                    last = FetchError(f"HTTP {reply.status_code}")
                    continue
                if reply.status_code >= 400:
                    raise FetchError(
                        f"{url} refused the request: HTTP {reply.status_code}"
                    )
                if span and reply.status_code != httpx.codes.PARTIAL_CONTENT:
                    raise FetchError(f"{url} ignored the byte range it was asked for")
                archive.answered()
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
            if isinstance(error, CONNECT_ERRORS):
                archive.refused()
            last = error
    raise FetchError(f"gave up after {deadline:.0f}s or {retries} retries: {last}")
