"""Bringing down some byte ranges of a file, each written where it sits in it."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import httpx

from common.disk.files import atomic_path
from common.fetch import http

BATCH = 600

CONTENT_RANGE = re.compile(rb"content-range:\s*bytes\s+(\d+)-(\d+)", re.IGNORECASE)


def parts(body: bytes, asked: Sequence[tuple[int, int]]) -> Iterator[tuple[int, bytes]]:
    """Read every part one ranged reply holds, with where it sits in the file.

    Args:
        body: The reply as it came, multipart when it answers more than one range.
        asked: The ranges the reply was asked for.

    Yields:
        part: The first byte of the file it holds, and its bytes.

    Raises:
        FetchError: When several ranges were asked and the reply is not in parts.
    """
    if len(asked) == 1:
        yield asked[0][0], body
        return
    boundary = body.lstrip(b"\r\n").split(b"\r\n", 1)[0]
    if not boundary.startswith(b"--"):
        raise http.FetchError("the archive did not answer the ranges in parts")
    at = body.index(boundary)
    # Each part is read by the length it declares, so its bytes are never searched.
    while not body.startswith(b"--", at := at + len(boundary)):
        head = body.index(b"\r\n\r\n", at)
        first, last = map(int, CONTENT_RANGE.search(body, at, head).groups())
        start = head + 4
        yield first, body[start : start + last - first + 1]
        at = body.index(boundary, start + last - first + 1)


def patched(
    url: str,
    path: Path,
    spans: Sequence[tuple[int, int]],
    size: int,
    timeout: float,
    *,
    client: httpx.Client | None = None,
    origin: int = 0,
) -> None:
    """Write some byte ranges of one file into a copy that leaves the rest a hole.

    Args:
        url: Where to read the file from.
        path: Where the copy belongs.
        spans: The first and past-the-last byte of each range, in order.
        size: How many bytes the copy holds.
        timeout: How long to wait on one transfer, between one chunk and the next.
        client: A client whose connections to reuse, or None to open one each.
        origin: Which byte of the file the copy starts at.

    Raises:
        FetchError: When refused, or answered with fewer bytes than were asked.
    """
    # Touching ranges are asked as one, so no server joins them into a plain reply.
    joined: list[tuple[int, int]] = []
    for first, last in spans:
        if joined and first <= joined[-1][1]:
            joined[-1] = (joined[-1][0], max(joined[-1][1], last))
        else:
            joined.append((first, last))
    with atomic_path(path) as tmp, tmp.open("r+b") as copy:
        copy.truncate(size)
        body = tmp.with_suffix(".part")
        try:
            # One request at a time, so a copy never opens more connections than a file.
            for start in range(0, len(joined), BATCH):
                batch = joined[start : start + BATCH]
                http.streamed(url, body, timeout, client=client, spans=batch)
                held = 0
                for first, data in parts(body.read_bytes(), batch):
                    copy.seek(first - origin)
                    copy.write(data)
                    held += len(data)
                if held < sum(last - first for first, last in batch):
                    raise http.FetchError(f"{url} sent fewer bytes than were asked")
        finally:
            body.unlink(missing_ok=True)
