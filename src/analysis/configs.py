"""Reading the analysis config, the one place a run is settled from."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from analysis import paths
from analysis.models.instrument import InstrumentSet
from analysis.models.settings import Settings


def load(path: Path = paths.CONFIG_PATH, workers: int | None = None) -> Settings:
    """Settle what a run should do, reading the config file once.

    Args:
        path: The config file, which carries every setting a run turns on.
        workers: How many jobs each half runs at once, standing in for the
            config where a run was given a number of cores of its own.

    Returns:
        choices: The settled choices for the run.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    plotted = config.get("plot_instruments")
    workers = workers or config["workers"]
    return Settings(
        grid_cells=config["grid_cells"],
        instrument_sets=tuple(
            InstrumentSet.from_key(key) for key in config["instruments"]
        ),
        plot_instrument_sets=tuple(InstrumentSet.from_key(key) for key in plotted)
        if plotted
        else None,
        loc=config["loc"],
        refresh_catalog=config["refresh_catalog"],
        workers=workers,
        # The coverage jobs run side by side, so each takes a share of the machine
        union_threads=max(1, (os.process_cpu_count() or 1) // workers),
    )
