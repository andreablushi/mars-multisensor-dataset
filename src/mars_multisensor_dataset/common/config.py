"""Every config file, read through Hydra into the model it settles."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mars_multisensor_dataset.analysis.models.settings import AnalysisSettings
from mars_multisensor_dataset.building.models.settings import (
    BuildSettings,
    TrainingSettings,
)
from mars_multisensor_dataset.common.paths import CONFIGS_ROOT


@cache
def load_config[T](
    name: str,
    model: type[T],
    overrides: tuple[str, ...] = (),
    root: Path = CONFIGS_ROOT,
    **values: object,
) -> T:
    """Read one config file, and every file its defaults compose, into its model.

    Args:
        name: The config file under the root, without its suffix.
        model: The dataclass it settles, whose fields every value is checked against.
        overrides: Hydra overrides of the file, such as `downloads.jpl=4`.
        root: The directory the file is read from; any other than the dataset's
            also searches the dataset's, so its defaults can compose `building`.
        **values: Values standing in for the file's own, each left out where None.

    Returns:
        settled: The model, read once a run whatever asks for it again.
    """
    searched = (
        () if root == CONFIGS_ROOT else (f"hydra.searchpath=[file://{CONFIGS_ROOT}]",)
    )
    with initialize_config_dir(str(root), version_base=None):
        read = compose(name, overrides=[*overrides, *searched])
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
