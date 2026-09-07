"""Reading the building runner config, the one place a build is configured from."""

from __future__ import annotations

from pathlib import Path

import yaml

import utils.disk.paths as paths
from building.models.settings import Settings


def load(path: Path = paths.BUILDING_CONFIG_PATH, cores: int | None = None) -> Settings:
    """Settle what a build should do, reading the config file once.

    Args:
        path: The config file, which carries every setting a build turns on.
        cores: How many cores the run was given, for a job a platform sized
            itself, and None to read the machine's own.

    Returns:
        The settled choices for the build.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Settings(
        name=config["name"],
        share=config["share"],
        seed=config["seed"],
        cores=cores,
    )
