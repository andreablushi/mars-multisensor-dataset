"""How bytes reach disk, and come back."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary path that replaces the destination on success.

    Args:
        path: The destination the temporary file is renamed to.

    Yields:
        path: The temporary path to write to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    tmp = Path(raw)
    try:
        yield tmp
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write rows as JSONL, atomically via a temp file and rename.

    Args:
        path: Destination file path.
        rows: An iterable of JSON serialisable mappings.

    Returns:
        rows: The number of rows written.
    """
    count = 0
    with atomic_path(path) as tmp, tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON object on each non empty line of a JSONL file.

    Args:
        path: The JSONL file to read.

    Yields:
        held: One decoded object per line.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
