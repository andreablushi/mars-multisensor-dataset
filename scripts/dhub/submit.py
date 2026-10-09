"""Registering a version of one stage on DigitalHub, and starting it."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import digitalhub as dh
from omegaconf import OmegaConf

from building.preprocessing.ctx import isis
from common.paths import CONFIGS_ROOT

STAGES = {
    "pipeline": "scripts.analysis_pipeline:run_pipeline",
    "selection": "scripts.analysis_pipeline:run_selection",
    "build_training": "scripts.build_training:run_build",
    "build_evaluation": "scripts.build_evaluation:run_build",
    "convert": "scripts.convert_archives:run_conversion",
}

REBUILT = (
    "build the dataset again from nothing, rather than filling in what the last "
    "build left missing"
)


def script_parser(description: str, forced: str = REBUILT) -> argparse.ArgumentParser:
    """Return the parser every script shares, for it to add its own flags to.

    Args:
        description: What the script does, shown by --help.
        forced: What --force redoes in this script.

    Returns:
        parser: The parser holding --dh, --force and --ref.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dh", action="store_true", help="submit to DigitalHub instead of running here"
    )
    parser.add_argument("--force", action="store_true", help=forced)
    parser.add_argument("--ref", default="main", help="branch, tag, or commit to run")
    return parser


def submitted(stage: str, ref: str, overrides: Sequence[str] = (), **parameters) -> int:
    """Register a version of one stage from a pushed commit, and run it.

    Args:
        stage: The stage to submit, as `digitalhub.yaml` names its resources.
        ref: The branch, tag, or commit the platform clones.
        overrides: Hydra overrides, `resources.` ones of the platform config and the
            rest of the stage's own, handed to the job.
        **parameters: What the handler is called with on the platform.

    Returns:
        code: A process exit code, zero once the job is started.
    """
    platform = OmegaConf.merge(
        OmegaConf.load(CONFIGS_ROOT / "digitalhub.yaml"),
        OmegaConf.from_dotlist(
            [one for one in overrides if one.startswith("resources.")]
        ),
    )
    handed = [one for one in overrides if not one.startswith("resources.")]
    asked = platform.resources[stage]
    # The job installs the clone's requirements.txt at start, so no image is built
    function = dh.get_or_create_project(platform.project).new_function(
        name=stage.replace("_", "-"),
        kind="python",
        python_version=platform.python_version,
        base_image=platform.base_image,
        code_src=f"git+{platform.repository}#{ref}",
        handler=STAGES[stage],
    )

    # Start the job, told where the clone lands and what the box holds
    root = platform.source_root
    run = function.run(
        action="job",
        profile=asked.profile + ("-shared" if asked.get("shared") else ""),
        resources={"cpu": str(asked.cpu), "mem": asked.memory, "disk": asked.disk},
        secrets=["DHCORE_PERSONAL_ACCESS_TOKEN"],
        envs=[
            {"name": "PYTHONPATH", "value": f"{root}:{root}/src:{root}/scripts"},
            *(isis.ENVS if asked.get("isis") else []),
        ],
        parameters=parameters
        | {"workers": asked.cpu}
        | ({"overrides": handed} if handed else {}),
        wait=False,
    )
    print(run.key)
    return 0
