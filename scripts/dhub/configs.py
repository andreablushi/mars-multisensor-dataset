"""Reading `configs/digitalhub.yaml`, the one file a platform run is settled from."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from shared.paths import CONFIGS_ROOT

PLATFORM_CONFIG_PATH = CONFIGS_ROOT / "digitalhub.yaml"


@dataclass(frozen=True, slots=True)
class Platform:
    """What a run submitted to DigitalHub is given, and what it publishes.

    Attributes:
        project: The project every run and every published archive belongs to.
        repository: The repository the platform clones to build the image.
        source_root: Where that clone lands on the job.
        python_version: The interpreter the image is built on.
        image_extras: What the platform itself asks for, beyond the pipeline.
        resources: The cores, the memory, the budget it plans against and the disk
            each stage asks for, by stage.
        functions: The function each stage is registered as, by stage.
        publishes: What each stage publishes, by the name a download asks for.
    """

    project: str
    repository: str
    source_root: str
    python_version: str
    image_extras: list[str]
    resources: dict[str, dict[str, str]]
    functions: dict[str, str]
    publishes: dict[str, str]


@lru_cache(maxsize=1)
def load(path: Path = PLATFORM_CONFIG_PATH) -> Platform:
    """Settle what a platform run is given, reading the config file once.

    Args:
        path: The config file, which carries every setting a run is submitted with.

    Returns:
        platform: The settled choices for the submission.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    asked = {
        stage: {key: str(value) for key, value in one.items()}
        for stage, one in config["resources"].items()
    }
    return Platform(**config | {"resources": asked})
