"""How bytes reach disk, and come back."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows as JSONL, atomically via a temp file and rename.

    Args:
        path: Destination file path.
        rows: An iterable of JSON serialisable mappings.
    """
    with atomic_path(path) as tmp, tmp.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON object on each non empty line of a JSONL file.

    Args:
        path: The JSONL file to read.

    Yields:
        held: One decoded object per line.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if stripped := line.strip():
                yield json.loads(stripped)


def write_json(path: Path, held: Any, end: str = "", **layout: Any) -> None:
    """Write one JSON value, atomically via a temp file and rename.

    Args:
        path: Destination file path.
        held: A JSON serialisable value.
        end: What follows the value, such as a closing newline.
        **layout: What `json.dumps` is handed, such as indent or default.
    """
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(held, **layout) + end, encoding="utf-8")


def read_json(path: Path) -> Any:
    """Return the JSON value one file holds."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write named arrays compressed, atomically via a temp file and rename.

    Args:
        path: Destination file path.
        arrays: The arrays to store, by the name each is stored under.
    """
    with atomic_path(path) as tmp, tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
