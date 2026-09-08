"""Reading the building config, the one place a build is settled from."""

from __future__ import annotations

from pathlib import Path

import yaml

from building import paths
from building.models.settings import Settings


def load(path: Path = paths.CONFIG_PATH, workers: int | None = None) -> Settings:
    """Settle what a build should do, reading the config file once.

    Args:
        path: The config file, which carries every setting a build turns on.
        workers: How many products to build at once, standing in for the config
            where a run was given a number of cores of its own.

    Returns:
        choices: The settled choices for the build.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Settings(
        name=config["name"],
        share=config["share"],
        seed=config["seed"],
        max_observations=config["max_observations"],
        workers=workers or config["workers"],
        downloads=config["downloads"],
    )
