"""The analysis config, read into the model it settles."""

from __future__ import annotations

from mars_multisensor_dataset.analysis.models.settings import AnalysisSettings
from mars_multisensor_dataset.common.config import load_config


def analysis_settings(workers: int | None = None) -> AnalysisSettings:
    """Read `analysis.yaml`, its workers overridden unless None."""
    return load_config("analysis", AnalysisSettings, workers=workers)
