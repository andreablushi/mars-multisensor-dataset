"""Registering a version of the pipeline on DigitalHub, and starting it."""

from __future__ import annotations

import tomllib

import digitalhub as dh

from building.preprocessing.ctx import isis
from common import paths
from dhub import configs, credentials
from dhub.artifacts import Function


def submitted(stage: Function, ref: str, **parameters) -> int:
    """Register a version of one stage from a pushed commit, and run it.

    Args:
        stage: The stage to submit, naming its function, handler and resources.
        ref: The branch, tag, or commit the platform clones.
        **parameters: What the handler is called with on the platform.

    Returns:
        code: A process exit code, non zero when the image did not build.
    """
    platform = configs.load()
    # The image is built from the repo's own dependencies, so it cannot drift
    manifest = (paths.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    needs = tomllib.loads(manifest)["project"]["dependencies"] + platform.image_extras
    asked = platform.resources[stage.name.lower()]
    project = dh.get_or_create_project(platform.project)
    function = project.new_function(
        name=stage.registered,
        kind="python",
        python_version=platform.python_version,
        code_src=f"git+{platform.repository}#{ref}",
        handler=stage.value,
        requirements=needs,
    )

    # Build the image first, since the job cannot install anything itself.
    built = function.run(
        action="build",
        profile=platform.resources["image"].profile,
        # Only the builds calibrate CTX, so only their image carries ISIS
        instructions=isis.INSTRUCTIONS if asked.isis else [],
        wait=True,
    )
    if built.status.state != "COMPLETED":
        print(f"the image did not build: {built.status.state}")
        return 1
    function.refresh()

    # Start the job, told where the clone lands and what the box holds
    root = platform.source_root
    run = function.run(
        action="job",
        profile=asked.profile,
        resources={"cpu": str(asked.cpu), "mem": asked.memory, "disk": asked.disk},
        secrets=[credentials.TOKEN],
        envs=[
            {"name": "PYTHONPATH", "value": f"{root}:{root}/src:{root}/scripts"},
            *credentials.minting_envs(),
            *(isis.ENVS if asked.isis else []),
        ],
        parameters=parameters | {"workers": asked.cpu},
        wait=False,
    )
    print(run.key)
    return 0
