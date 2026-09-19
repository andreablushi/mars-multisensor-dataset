"""Reading the building config, the one place a build is settled from."""

from __future__ import annotations

from pathlib import Path

import yaml

from common.building import paths
from common.building.models.settings import Settings


def load(
    name: str, path: Path = paths.CONFIG_PATH, workers: int | None = None
) -> Settings:
    """Settle how a build runs, whichever dataset it builds, reading the config once.

    Args:
        name: What the dataset built is called, which its own config names.
        path: The config file, which carries how every build runs.
        workers: How many products to build at once, standing in for the config
            where a run was given a number of cores of its own.

    Returns:
        choices: The settled choices for the build.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Settings(
        name=name,
        workers=workers or config["workers"],
        downloads=config["downloads"],
    )
