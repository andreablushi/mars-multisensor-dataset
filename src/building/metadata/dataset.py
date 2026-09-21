"""What the whole dataset is, which no row of it can say for itself."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime

from analysis import paths as analysis_paths
from common import paths


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """What the dataset is and what it was built from.

    Attributes:
        built_at: When the build that wrote it finished, in UTC.
        instruments: The instruments it holds crops of.
        selection: Where the selection it was built from was read.
        revision: The commit the build ran from, or None outside a checkout.
        band_centres_nm: The nominal band centres in nm, by instrument.
    """

    built_at: str
    instruments: tuple[str, ...]
    selection: str
    revision: str | None
    band_centres_nm: dict[str, tuple[float, ...]]


def dataset_manifest(
    instruments: tuple[str, ...], band_centres_nm: dict[str, tuple[float, ...]]
) -> DatasetManifest:
    """Return what to write beside the dataset to say what it is.

    Args:
        instruments: The instruments the build covered.
        band_centres_nm: The band grid of each instrument with a wavelength.

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
    return DatasetManifest(
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        instruments=tuple(sorted(instruments)),
        selection=str(analysis_paths.SELECTION_ROOT.relative_to(paths.REPO_ROOT)),
        revision=revision,
        band_centres_nm={
            name: band_centres_nm[name] for name in sorted(band_centres_nm)
        },
    )
