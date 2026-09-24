"""Everything a platform run publishes or reads back, and where each lands on disk."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from analysis import paths
from building import paths as building_paths


class Artifact(Enum):
    """One artifact on DigitalHub, its value where it lands on disk."""

    COVERAGE = paths.COVERAGE_ROOT
    METADATA = paths.METADATA_ROOT
    SELECTION = paths.SELECTION_ROOT
    STATS = paths.STATS_ROOT
    LABELS = paths.LABELS_ROOT
    SUMMARY = paths.COVERAGE_ROOT / paths.SUMMARY_NAME
    VERDICTS = paths.VERDICTS_PATH
    DATASET = building_paths.DATASETS_ROOT

    @property
    def published(self) -> str:
        """Return the name it is published under.

        Returns:
            name: Its member's name, in lower case.
        """
        return self.name.lower()

    @property
    def path(self) -> Path:
        """Return where it lands on disk.

        Returns:
            path: A directory for an archive, or the file itself.
        """
        return self.value

    @property
    def packed(self) -> bool:
        """Say whether it goes up as one archive of a directory.

        Returns:
            packed: True for a directory, False for a file published as it is.
        """
        return not self.path.suffix
