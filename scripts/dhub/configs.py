"""Reading `configs/digitalhub.yaml`, which settles every platform run."""

from __future__ import annotations

from dataclasses import dataclass, replace

from common.config import load_config
from common.paths import CONFIGS_ROOT

PLATFORM_CONFIG_PATH = CONFIGS_ROOT / "digitalhub.yaml"


@dataclass(slots=True)
class Resources:
    """What one stage asks the platform for.

    Attributes:
        profile: The profile settling the cores and the memory of the box.
        cpu: The cores the job is scheduled on, or None for a stage only built.
        memory: The memory it is scheduled with, such as "32Gi".
        disk: The disk it is given.
        budget: The memory a build plans against, or None to plan against all of it.
        shared: Whether it queues on the shared pool rather than the reserved one.
    """

    profile: str
    cpu: int | None = None
    memory: str | None = None
    disk: str | None = None
    budget: str | None = None
    shared: bool = False


@dataclass(slots=True)
class Platform:
    """What a run submitted to DigitalHub is given, and what it publishes.

    Attributes:
        project: The project every run and every published archive belongs to.
        repository: The repository the platform clones to build the image.
        source_root: Where that clone lands on the job.
        python_version: The interpreter the image is built on.
        image_extras: What the platform itself asks for, beyond the pipeline.
        resources: The profile, cores, memory, budget and disk of each stage.
        functions: The function each stage is registered as, by stage.
        publishes: What each stage publishes, by the name a download asks for.
    """

    project: str
    repository: str
    source_root: str
    python_version: str
    image_extras: list[str]
    resources: dict[str, Resources]
    functions: dict[str, str]
    publishes: dict[str, str]


def load() -> Platform:
    """Settle what a platform run is given, reading the config file once.

    Returns:
        platform: The settled choices, each profile marked for its pool.
    """
    platform = load_config(PLATFORM_CONFIG_PATH, Platform)
    return replace(
        platform,
        resources={
            stage: replace(one, profile=one.profile + ("-shared" if one.shared else ""))
            for stage, one in platform.resources.items()
        },
    )
