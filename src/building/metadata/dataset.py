"""What the whole dataset is, which no row of it can say for itself."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import utils.disk.paths as paths


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """What the dataset is and what it was built from.

    Attributes:
        version: Which layout the arrays and the index are written in.
        built_at: When the build that wrote it finished, in UTC.
        instruments: The instruments it holds crops of.
        selection: Where the selection it was built from was read.
        revision: The commit the build ran from, or None outside a checkout.
    """

    version: int
    built_at: str
    instruments: tuple[str, ...]
    selection: str
    revision: str | None


def dataset_manifest(instruments: tuple[str, ...], version: int) -> DatasetManifest:
    """Return what to write beside the dataset to say what it is.

    Args:
        instruments: The instruments the build covered.
        version: Which layout what is written is in, as the build was configured.

    Returns:
        The manifest, its revision unset where the build ran outside a checkout.
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
        version=version,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        instruments=tuple(sorted(instruments)),
        selection=str(paths.SELECTION_ROOT.relative_to(paths.REPO_ROOT)),
        revision=revision,
    )


def as_written(manifest: DatasetManifest) -> dict:
    """Return the manifest as the object it is written as.

    Args:
        manifest: What the build has to say about the dataset.

    Returns:
        Its fields, ready to be written as JSON.
    """
    return asdict(manifest)
