"""Reading the build configs, one per dataset a build makes."""

from __future__ import annotations

from building.models.settings import Settings, TrainingSettings
from common.config import load_config
from common.paths import CONFIGS_ROOT

TRAINING_CONFIG_PATH = CONFIGS_ROOT / "building_training.yaml"
EVALUATION_CONFIG_PATH = CONFIGS_ROOT / "building_evaluation.yaml"


def training_settings(workers: int | None = None) -> TrainingSettings:
    """Settle what the training build should do, reading its config once.

    Args:
        workers: How many products to build at once, or None for the config.

    Returns:
        settings: The settled choices for the training build.
    """
    return load_config(TRAINING_CONFIG_PATH, TrainingSettings, workers=workers)


def evaluation_settings(workers: int | None = None) -> Settings:
    """Settle what the evaluation build should do, reading its config once.

    Args:
        workers: How many products to build at once, or None for the config.

    Returns:
        settings: The settled choices for the evaluation build.
    """
    return load_config(EVALUATION_CONFIG_PATH, Settings, workers=workers)
