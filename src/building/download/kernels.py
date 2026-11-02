"""The MRO kernels a CTX scan is placed with, fetched from NAIF as scans need them."""

from __future__ import annotations

import functools
import re
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from building.models.kernel import KernelSelection
from building.preprocessing.ctx.isis import DATA
from common.fetch import http

KERNELS = Path(DATA) / "mro" / "kernels"

NAIF = "https://naif.jpl.nasa.gov/pub/naif/MRO/kernels/"

KINDS = ("ck", "spk")

SUFFIXES = frozenset({".bc", ".bsp"})

TDB_FORMAT = "%Y %b %d %H:%M:%S.%f"

PAD = timedelta(minutes=5)

HELD_BYTES = 10 * 1024**3

SELECTION = re.compile(r"Group = Selection(.*?)End_Group", re.S)

TIME = re.compile(r'Time\s*=\s*\("([^"]+) TDB",\s*"([^"]+) TDB"\)')

FILE = re.compile(r"File\s*=\s*\$mro/kernels/(\S+)")

RECONSTRUCTED = "Type = Reconstructed"

_lock = threading.Lock()

_users: Counter[Path] = Counter()

_fetching: dict[Path, threading.Lock] = {}


@functools.cache
def read_kernel_selections(kind: str) -> tuple[KernelSelection, ...]:
    """Return every reconstructed kernel of one kind the newest database selects.

    Args:
        kind: The kernel directory under the mission, ck or spk.

    Returns:
        selections: Each reconstructed kernel and the spans it covers.
    """
    database = sorted((KERNELS / kind).glob("kernels.????.db"))[-1]
    return tuple(
        KernelSelection(
            KERNELS / name,
            NAIF + name,
            tuple(
                (
                    datetime.strptime(first, TDB_FORMAT),
                    datetime.strptime(last, TDB_FORMAT),
                )
                for first, last in TIME.findall(group)
            ),
        )
        for group in SELECTION.findall(database.read_text())
        if RECONSTRUCTED in group
        for name in FILE.findall(group)
    )


@contextmanager
def held_kernels(start: datetime, client: httpx.Client) -> Iterator[None]:
    """Hold every kernel a scan may be placed with on disk, fetching what is missing.

    Args:
        start: When the scan began, in UTC.
        client: The client every kernel is downloaded over.
    """
    needed = [
        one
        for kind in KINDS
        for one in read_kernel_selections(kind)
        if one.covers(start - PAD, start + PAD)
    ]
    with _lock:
        _users.update(one.path for one in needed)
        fetching = [_fetching.setdefault(one.path, threading.Lock()) for one in needed]
    try:
        for one, guard in zip(needed, fetching, strict=True):
            with guard:
                if not one.path.exists():
                    http.streamed(one.url, one.path, client=client, spans=())
            one.path.touch()
        yield
    finally:
        with _lock:
            _users.subtract(one.path for one in needed)
            held = sorted(
                (
                    path
                    for kind in KINDS
                    for path in (KERNELS / kind).iterdir()
                    if path.suffix in SUFFIXES
                ),
                key=lambda path: path.stat().st_mtime,
            )
            excess = sum(path.stat().st_size for path in held) - HELD_BYTES
            for path in held:
                if excess <= 0:
                    break
                if _users[path] <= 0:
                    excess -= path.stat().st_size
                    path.unlink()
