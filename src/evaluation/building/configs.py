"""Reading the evaluation building config, which names the build."""

from __future__ import annotations

from pathlib import Path

import yaml

from evaluation import paths


def load_name(path: Path = paths.BUILDING_CONFIG_PATH) -> str:
    """Read what the evaluation build is called.

    Args:
        path: The config file, which names the build.

    Returns:
        name: The directory it is written in and the name it is published under.
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))["name"]
