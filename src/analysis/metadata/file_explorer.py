"""What the downloaded metadata tree holds."""

from __future__ import annotations

from pathlib import Path

from analysis import paths


def find_sets() -> list[Path]:
    """Find every non-empty metadata file, sorted, one per group and instrument set."""
    return sorted(
        path
        for path in paths.METADATA_ROOT.glob("*/*.jsonl")
        if path.stat().st_size > 0
    )
