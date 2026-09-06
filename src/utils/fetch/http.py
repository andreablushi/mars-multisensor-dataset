"""How a request reaches a server that answers slowly, or does not answer at first."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from utils.disk.files import atomic_path

REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 20
BACKOFF_BASE = 0.5
# Ceiling on one backoff sleep, so many retries stay minutes rather than days
BACKOFF_MAX = 30.0
RETRYABLE_STATUS = frozenset({403, 429, 500, 502, 503, 504})
# Fewer tries for a transfer than for a query, since one runs for minutes and
# a job retrying every one of them would hang for hours
STREAM_RETRIES = 5


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

    Returns:
        What `accepted` read out of the first usable reply.

    Raises:
        FetchError: When the server refuses the request, or when no attempt
            left a reply `accepted` could read.
    """
    asking = client or httpx
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            slept(attempt, backoff)
        try:
            reply = asking.get(url, params=params, timeout=timeout)
        except httpx.HTTPError as error:
            last = error
            continue
        if reply.status_code in RETRYABLE_STATUS:
            last = FetchError(f"HTTP {reply.status_code}")
            continue
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
    raise FetchError(f"gave up after {retries} retries: {last}")


def streamed(
    url: str,
    path: Path,
    timeout: float,
    *,
    retries: int = STREAM_RETRIES,
    backoff: float = BACKOFF_BASE,
) -> None:
    """Stream one file to disk, asking again while the server keeps failing.

    Args:
        url: Where to read it from.
        path: Where it belongs once it is whole.
        timeout: How long to wait on one transfer.
        retries: How many times to ask again after the first attempt.
        backoff: The base delay between attempts, in seconds.

    Returns:
        None.

    Raises:
        FetchError: When the server refuses the file, or every attempt fails.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            slept(attempt, backoff)
        try:
            with httpx.stream("GET", url, timeout=timeout) as reply:
                if reply.status_code in RETRYABLE_STATUS:
                    last = FetchError(f"HTTP {reply.status_code}")
                    continue
                if reply.status_code >= 400:
                    raise FetchError(
                        f"{url} refused the request: HTTP {reply.status_code}"
                    )
                # Nothing is left behind when a transfer fails part way through.
                with atomic_path(path) as tmp, tmp.open("wb") as handle:
                    for chunk in reply.iter_bytes():
                        handle.write(chunk)
                return
        except httpx.HTTPError as error:
            last = error
    raise FetchError(f"gave up after {retries} retries: {last}")
