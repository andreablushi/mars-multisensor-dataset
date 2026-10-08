"""Every config file, read through Hydra into the model it settles."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from analysis.models.settings import AnalysisSettings
from building.models.settings import BuildSettings, TrainingSettings
from common.paths import CONFIGS_ROOT


@cache
def load_config[T](
    name: str, model: type[T], overrides: tuple[str, ...] = (), **values: object
) -> T:
    """Read one config file, and every file its defaults compose, into its model.

    Args:
        name: The config file under the configs root, without its suffix.
        model: The dataclass it settles, whose fields every value is checked against.
        overrides: Hydra overrides of the file, such as `downloads.jpl=4`.
        **values: Values standing in for the file's own, each left out where None.

    Returns:
        settled: The model, read once a run whatever asks for it again.
    """
    with initialize_config_dir(str(CONFIGS_ROOT), version_base=None):
        read = compose(name, overrides=list(overrides))
    given = {key: value for key, value in values.items() if value is not None}
    return OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(model), read, given)
    )


def analysis_settings(workers: int | None = None) -> AnalysisSettings:
    """Read `analysis.yaml`, its workers overridden unless None."""
    return load_config("analysis", AnalysisSettings, workers=workers)


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
