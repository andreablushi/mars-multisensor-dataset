"""Reading the evaluation analysis config, which every label is settled from."""

from __future__ import annotations

from pathlib import Path

import yaml

from evaluation import paths
from evaluation.analysis.models.rule import Rule
from evaluation.analysis.models.settings import Settings


def load(path: Path = paths.ANALYSIS_CONFIG_PATH) -> Settings:
    """Settle how the kept tiles are labelled, reading the config file once.

    Args:
        path: The config file, which carries every class and how it is read.

    Returns:
        choices: The settled choices for the labelling.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Settings(
        core=float(config["core"]),
        rules=tuple(
            Rule(
                label=label,
                descriptor=rule.get("descriptor"),
                names=tuple(rule.get("names", ())),
                diameter_km=tuple(rule["diameter_km"])
                if "diameter_km" in rule
                else None,
                latitudes=tuple(rule["latitudes"]) if "latitudes" in rule else None,
            )
            for label, rule in config["classes"].items()
        ),
    )
