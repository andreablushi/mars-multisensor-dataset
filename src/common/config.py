"""Reading a config file through Hydra into the model it settles, checked against it."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from common.paths import CONFIGS_ROOT


@cache
def load_config[T](path: Path, model: type[T], **overrides: object) -> T:
    """Read one config file, and every file its defaults compose, into its model.

    Args:
        path: The config file, under the configs root.
        model: The dataclass it settles, whose fields every value is checked against.
        **overrides: Values standing in for the file's own, each left out where None.

    Returns:
        settled: The model, read once a run whatever asks for it again.
    """
    name = path.relative_to(CONFIGS_ROOT).with_suffix("")
    with initialize_config_dir(str(CONFIGS_ROOT), version_base=None):
        read = compose(name.as_posix())
    given = {key: value for key, value in overrides.items() if value is not None}
    return OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(model), read, given)
    )
