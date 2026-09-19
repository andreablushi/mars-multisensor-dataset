"""Reading the analysis config, the one place a run is settled from."""

from __future__ import annotations

import os
from dataclasses import replace

from analysis import paths
from analysis.models.settings import Settings
from common.config import load_config


def load(workers: int | None = None) -> Settings:
    """Settle what a run should do, reading the config file once.

    Args:
        workers: How many jobs each half runs at once, standing in for the
            config where a run was given a number of cores of its own, which
            are then every core the jobs share.

    Returns:
        choices: The settled choices for the run.
    """
    settings = load_config(paths.CONFIG_PATH, Settings, workers=workers)
    # The coverage jobs run side by side, so each takes a share of the machine
    cores = workers or os.process_cpu_count() or 1
    return replace(settings, union_threads=max(1, cores // settings.workers))
