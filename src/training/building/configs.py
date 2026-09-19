"""Reading the training building config, which settles what the build draws."""

from __future__ import annotations

from pathlib import Path

import yaml

from training import paths
from training.building.models.settings import Settings


def load(path: Path = paths.BUILDING_CONFIG_PATH) -> Settings:
    """Settle what the training build draws, reading the config file once.

    Args:
        path: The config file, which carries every setting the draw turns on.

    Returns:
        choices: The settled choices for the build.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Settings(name=config["name"], share=config["share"], seed=config["seed"])
