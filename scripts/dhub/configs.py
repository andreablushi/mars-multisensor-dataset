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
        cpu: The cores the job is scheduled on.
        memory: The memory it is scheduled with, such as "32Gi".
        disk: The disk it is given.
        shared: Whether it queues on the shared pool rather than the reserved one.
        isis: Whether its job installs ISIS at start and is told where.
    """

    profile: str
    cpu: int
    memory: str
    disk: str
    shared: bool = False
    isis: bool = False


@dataclass(slots=True)
class Platform:
    """What a run submitted to DigitalHub is given.

    Attributes:
        project: The project every run and every published archive belongs to.
        repository: The repository the platform clones when a job starts.
        source_root: Where that clone lands on the job.
        python_version: The interpreter a job runs on.
        base_image: The platform's own base image a job runs on.
        resources: The profile, cores, memory and disk of each stage.
    """

    project: str
    repository: str
    source_root: str
    python_version: str
    base_image: str
    resources: dict[str, Resources]


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
