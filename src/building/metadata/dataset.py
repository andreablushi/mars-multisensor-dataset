"""What the whole dataset is, which no row of it can say for itself."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis import paths as analysis_paths
from building import paths as built
from building.models.instrument import INSTRUMENTS
from common import paths
from common.disk.files import read_json, write_json


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """What the dataset is and what it was built from.

    Attributes:
        built_at: When the build that wrote it finished, in UTC.
        instruments: The instruments it holds crops of.
        selection: Where the selection it was built from was read.
        revision: The commit the build ran from, or None outside a checkout.
        band_centres_nm: The nominal band centres in nm, by instrument.
        normalization: The mean and std each instrument is standardised by, per band
            where it has bands, or None before the dataset is normalized.
    """

    built_at: str
    instruments: tuple[str, ...]
    selection: str
    revision: str | None
    band_centres_nm: dict[str, tuple[float, ...]]
    normalization: dict[str, dict[str, Any]] | None = None


def dataset_manifest(
    instruments: Iterable[str], normalization: dict[str, dict[str, Any]] | None
) -> DatasetManifest:
    """Return what to write beside the dataset to say what it is.

    Args:
        instruments: The instruments the dataset holds crops of.
        normalization: The constants it is standardised by, or None.

    Returns:
        manifest: The manifest, its revision unset outside a checkout.
    """
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    held = tuple(sorted(instruments))
    return DatasetManifest(
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        instruments=held,
        selection=str(analysis_paths.SELECTION_ROOT.relative_to(paths.REPO_ROOT)),
        revision=revision,
        band_centres_nm={
            name: INSTRUMENTS[name].layout.band_centres_nm
            for name in held
            if name in INSTRUMENTS and INSTRUMENTS[name].layout.band_centres_nm
        },
        normalization=normalization,
    )


def read_normalization(root: Path) -> dict[str, dict[str, Any]] | None:
    """Return the constants one build's manifest records, or None where it has none.

    Args:
        root: The directory the build was written in.

    Returns:
        normalization: The mean and std of each instrument, or None.
    """
    manifest = read_manifest(root)
    return None if manifest is None else manifest.normalization


def read_manifest(root: Path) -> DatasetManifest | None:
    """Return the manifest one build wrote, or None where it wrote none.

    Args:
        root: The directory the build was written in.

    Returns:
        manifest: The manifest as written, or None.
    """
    path = root / built.DATASET_MANIFEST_NAME
    return DatasetManifest(**read_json(path)) if path.exists() else None


def write_manifest(manifest: DatasetManifest, root: Path) -> None:
    """Write a build's manifest beside its index, atomically.

    Args:
        manifest: What the dataset is.
        root: The directory the build is written in.
    """
    write_json(root / built.DATASET_MANIFEST_NAME, asdict(manifest), indent=2)
