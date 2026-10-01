"""How a request reaches a server that answers slowly, or does not answer at first."""

from __future__ import annotations

import functools
import random
import ssl
import time
from collections.abc import Callable, Iterator, Sequence
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
CROWDED_STATUS = frozenset({403, 429, 503})
# Fewer tries for a transfer than a query, one running for minutes not seconds
STREAM_RETRIES = 5
# How long to wait for the larger half of a product.
STREAM_TIMEOUT = 60.0

CONNECT_TIMEOUT = 30.0

CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)

# How long one query may be asked for in all, an attempt count bounding nothing
QUERY_DEADLINE = 300.0

# How long one transfer may run in all, so a trickling server is given up on
STREAM_DEADLINE = 1800.0

TLS_CONTEXT = httpx.create_ssl_context()
TLS_CONTEXT.verify_flags &= ~ssl.VERIFY_X509_STRICT


@functools.cache
def throttle(host: str) -> Throttle:
    """Return the throttle one host's requests share, held back by its refusals."""
    return Throttle()


class FetchError(RuntimeError):
    """Raised when a server refuses a request, or keeps failing to answer one."""


def gave_up(host: str, started: float, last: Exception) -> FetchError:
    """Return the error a fetch ends on, naming the host and what it last failed with.

    Args:
        host: The host that was asked.
        started: When the fetch began, on the monotonic clock.
        last: What the last attempt failed with.

    Returns:
        error: The error to raise.
    """
    elapsed = time.monotonic() - started
    return FetchError(
        f"gave up on {host} after {elapsed:.0f}s, {type(last).__name__}: {last}"
    )


def transport_failure(
    error: httpx.HTTPError, host: str, archive: Throttle
) -> httpx.HTTPError:
    """Return what an attempt failed with, to ask again unless a certificate refused it.

    Args:
        error: What the attempt failed with.
        host: The host that was asked.
        archive: The throttle of that host, held back when it could not be reached.

    Returns:
        error: The same error, named should every attempt fail.

    Raises:
        FetchError: When a certificate check lies anywhere in its chain.
    """
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, ssl.SSLCertVerificationError):
            raise FetchError(f"{host} failed its certificate check: {error}") from error
        cause = cause.__cause__ or cause.__context__
    if isinstance(error, CONNECT_ERRORS):
        archive.refused()
    return error


def retry_error(status: int, url: str, archive: Throttle) -> FetchError | None:
    """Return the error to ask again on, or None once the server has answered.

    Args:
        status: The HTTP status the server replied with.
        url: Where it was asked.
        archive: The throttle of its host, eased by an answer and held by a crowd.

    Returns:
        retry: The error to ask again on, or None for a reply worth reading.

    Raises:
        FetchError: When the server refuses the request outright.
    """
    if status in RETRYABLE_STATUS:
        # One refusal slows every thread on that host, so a run stops being blocked.
        if status in CROWDED_STATUS:
            archive.refused()
        return FetchError(f"HTTP {status}")
    archive.answered()
    if status >= 400:
        raise FetchError(f"{url} refused the request: HTTP {status}")
    return None


def attempts(archive: Throttle, give_up_at: float, retries: int) -> Iterator[None]:
    """Yield once per attempt, each waited out longer than the last and never in step.

    Args:
        archive: The throttle of the host asked, waited out before every attempt.
        give_up_at: When to stop asking, on the monotonic clock.
        retries: How many times to ask again after the first attempt.

    Yields:
        None: Once an attempt may be made.
    """
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(
                min(BACKOFF_BASE * 2 ** (attempt - 1), BACKOFF_MAX)
                + random.uniform(0.0, BACKOFF_BASE)
            )
        if time.monotonic() >= give_up_at:
            return
        archive.wait()
        yield


def fetched_json(
    url: str,
    params: dict[str, str],
    *,
    accepted: Callable[[Any], Any | None],
    client: httpx.Client,
) -> Any:
    """Read one JSON reply, asking again until the server answers a usable one.

    Args:
        url: Where to ask.
        params: What to ask for.
        accepted: What reads one reply, or hands back None to ask again.
        client: The client whose connections to reuse.

    Returns:
        found: What `accepted` read out of the first usable reply.

    Raises:
        FetchError: When refused, when no reply was readable, or past the deadline.
    """
    host = httpx.URL(url).host
    archive = throttle(host)
    started = time.monotonic()
    last: Exception = FetchError("no attempt was made")
    for _ in attempts(archive, started + QUERY_DEADLINE, MAX_RETRIES):
        try:
            reply = client.get(
                url,
                params=params,
                timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
            )
        except httpx.HTTPError as error:
            last = transport_failure(error, host, archive)
            continue
        if retry := retry_error(reply.status_code, url, archive):
            last = retry
            continue
        try:
            payload = reply.json()
        except ValueError as error:
            last = error
            continue
        found = accepted(payload)
        if found is not None:
            return found
        last = FetchError("the reply held nothing to read")
    raise gave_up(host, started, last)


def streamed(
    url: str,
    path: Path,
    *,
    client: httpx.Client,
    spans: Sequence[tuple[int, int]],
) -> None:
    """Stream one file to disk, asking again while the server keeps failing.

    Args:
        url: Where to read it from.
        path: Where it belongs once it is whole.
        client: The client whose connections to reuse.
        spans: The first and past-the-last byte of each part to keep, or none for all.

    Raises:
        FetchError: When refused, when every attempt fails, or past the deadline.
    """
    ranges = ",".join(f"{first}-{last - 1}" for first, last in spans)
    headers = {"Range": f"bytes={ranges}"} if spans else None
    host = httpx.URL(url).host
    archive = throttle(host)
    started = time.monotonic()
    last: Exception = FetchError("no attempt was made")
    give_up_at = started + STREAM_DEADLINE
    for _ in attempts(archive, give_up_at, STREAM_RETRIES):
        try:
            with client.stream(
                "GET",
                url,
                timeout=httpx.Timeout(STREAM_TIMEOUT, connect=CONNECT_TIMEOUT),
                headers=headers,
            ) as reply:
                if retry := retry_error(reply.status_code, url, archive):
                    last = retry
                    continue
                if spans and reply.status_code != httpx.codes.PARTIAL_CONTENT:
                    raise FetchError(f"{url} ignored the byte range it was asked for")
                # Nothing is left behind when a transfer fails part way through.
                with atomic_path(path) as tmp, tmp.open("wb") as handle:
                    for chunk in reply.iter_bytes():
                        # A timeout bounds one chunk, and this the whole transfer
                        if time.monotonic() >= give_up_at:
                            raise FetchError(
                                f"{url} was still sending after {STREAM_DEADLINE:.0f}s"
                            )
                        handle.write(chunk)
                return
        except httpx.HTTPError as error:
            last = transport_failure(error, host, archive)
    raise gave_up(host, started, last)
