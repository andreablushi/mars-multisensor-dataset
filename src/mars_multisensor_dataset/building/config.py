"""The build configs, each read into the model it settles."""

from __future__ import annotations

from collections.abc import Sequence

from mars_multisensor_dataset.building.models.settings import (
    BuildSettings,
    TrainingSettings,
)
from mars_multisensor_dataset.common.config import load_config


def training_settings(
    workers: int | None = None, overrides: Sequence[str] = ()
) -> TrainingSettings:
    """Read `building_training.yaml` with its overrides, its workers unless None."""
    return load_config(
        "building_training", TrainingSettings, tuple(overrides), workers=workers
    )


def evaluation_settings(
    workers: int | None = None, overrides: Sequence[str] = ()
) -> BuildSettings:
    """Read `building_evaluation.yaml` with its overrides, its workers unless None."""
    return load_config(
        "building_evaluation", BuildSettings, tuple(overrides), workers=workers
    )
