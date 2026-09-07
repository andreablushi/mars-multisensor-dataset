"""What the downloaded metadata tree holds."""

from __future__ import annotations

from pathlib import Path

import utils.disk.paths as paths
from analysis.models.instrument import InstrumentSet


def find_sets(root: Path = paths.METADATA_ROOT) -> list[Path]:
    """Find every stored instrument set holding observations.

    Args:
        root: The metadata root directory.

    Returns:
        files: The non-empty JSONL files, sorted, one per feature and instrument set.
    """
    return sorted(path for path in root.glob("*/*/*.jsonl") if path.stat().st_size > 0)


def has_metadata(
    feature_dir: Path, instrument_set: InstrumentSet, root: Path = paths.METADATA_ROOT
) -> bool:
    """Report whether one feature holds downloaded records for an instrument set.

    Args:
        feature_dir: The feature's artifacts directory
        instrument_set: The instrument set to look for.
        root: The metadata root directory.

    Returns:
        found: True when a non-empty metadata file for that set exists.
    """
    directory = root / feature_dir.parent.name / feature_dir.name
    return any(
        path.stat().st_size > 0
        for path in directory.glob(f"{instrument_set.slug}*.jsonl")
    )
