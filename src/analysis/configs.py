"""Reading the analysis config, the one place a run is settled from."""

from __future__ import annotations

import os
from dataclasses import replace

from analysis.models.settings import Settings
from common.config import load_config
from common.paths import CONFIGS_ROOT

CONFIG_PATH = CONFIGS_ROOT / "analysis.yaml"


def load(workers: int | None = None) -> Settings:
    """Settle what a run should do, reading the config file once.

    Args:
        workers: How many jobs each half runs at once, or None for the config.

    Returns:
        choices: The settled choices for the run.
    """
    settings = load_config(CONFIG_PATH, Settings, workers=workers)
    # The coverage jobs run side by side, so each takes a share of the machine
    cores = workers or os.process_cpu_count() or 1
    return replace(settings, union_threads=max(1, cores // settings.workers))
