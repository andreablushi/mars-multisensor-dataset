"""Where what a platform run publishes or reads back lands, and what it runs as."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from analysis import paths as analysis_paths
from building import paths as building_paths


class Artifact(Enum):
    """One artifact on DigitalHub, its value where it lands on disk."""

    COVERAGE = analysis_paths.COVERAGE_ROOT
    METADATA = analysis_paths.METADATA_ROOT
    SELECTION = analysis_paths.SELECTION_ROOT
    STATS = analysis_paths.STATS_ROOT
    LABELS = analysis_paths.LABELS_ROOT
    SUMMARY = analysis_paths.COVERAGE_ROOT / analysis_paths.SUMMARY_NAME
    VERDICTS = analysis_paths.VERDICTS_PATH
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


class Function(Enum):
    """One stage as DigitalHub registers it, its value the handler a job calls."""

    PIPELINE = "scripts.analysis_pipeline:run_pipeline"
    SELECTION = "scripts.analysis_pipeline:run_selection"
    BUILD_TRAINING = "scripts.build_training:run_build"
    BUILD_EVALUATION = "scripts.build_evaluation:run_build"

    @property
    def registered(self) -> str:
        """Return the name the function is registered under.

        Returns:
            name: Its member's name, in lower kebab case.
        """
        return self.name.lower().replace("_", "-")
