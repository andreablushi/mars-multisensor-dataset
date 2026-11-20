"""Running one stage as a platform job, the repository's requirements installed."""

from __future__ import annotations

import importlib
import subprocess
import sys

from digitalhub_runtime_python.utils.inputs import compose_inputs

UV = "/opt/nuclio/uv/uv"

STAGES = {
    "pipeline": "scripts.analysis_pipeline:run_pipeline",
    "selection": "scripts.analysis_pipeline:run_selection",
    "build_training": "scripts.build_training:run_build",
    "build_evaluation": "scripts.build_evaluation:run_build",
}


def job_parameters(stage: str, arguments: dict, requirements: list[str]) -> None:
    """Name what the platform hands a job, for its runtime to compose.

    Args:
        stage: Which stage, naming its handler.
        arguments: What the stage's handler is called with.
        requirements: The repository's pinned requirements.
    """


def run_installed_stage(
    project_name: str,
    run_key: str,
    stage: str,
    arguments: dict,
    requirements: list[str],
) -> dict:
    """Return the outputs of one stage's handler, the requirements installed first.

    Args:
        project_name: The project, as the runtime hands a handler.
        run_key: The run, as the runtime hands a handler.
        stage: Which stage, naming its handler.
        arguments: What the stage's handler is called with.
        requirements: The repository's pinned requirements.

    Returns:
        outputs: What the stage's handler logged, for the run's status.
    """
    subprocess.run(
        [UV, "pip", "install", "--python", sys.executable, *requirements], check=True
    )
    module, name = STAGES[stage].split(":")
    stage_handler = getattr(importlib.import_module(module), name)
    composed = compose_inputs({}, arguments, False, stage_handler, project_name)
    return stage_handler(project_name, run_key, **composed)


run_installed_stage.__wrapped__ = job_parameters
