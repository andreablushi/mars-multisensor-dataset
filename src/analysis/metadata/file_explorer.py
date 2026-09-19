"""What the downloaded metadata tree holds."""

from __future__ import annotations

from pathlib import Path

from analysis import paths
from analysis.models.instrument import InstrumentSet


def find_sets(root: Path = paths.METADATA_ROOT) -> list[Path]:
    """Find every stored instrument set holding observations.

    Args:
        root: The metadata root directory.

    Returns:
        files: The non-empty JSONL files, sorted, one per group and instrument set.
    """
    return sorted(path for path in root.glob("*/*.jsonl") if path.stat().st_size > 0)


def has_metadata(
    group: str, instrument_set: InstrumentSet, root: Path = paths.METADATA_ROOT
) -> bool:
    """Report whether one group holds downloaded records for an instrument set.

    Args:
        group: The name of the group.
        instrument_set: The instrument set to look for.
        root: The metadata root directory.

    Returns:
        found: True when a non-empty metadata file for that set exists.
    """
    held = paths.metadata_file(root, group, instrument_set)
    return held.exists() and held.stat().st_size > 0
