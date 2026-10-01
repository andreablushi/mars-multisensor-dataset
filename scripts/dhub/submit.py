"""Registering a version of the pipeline on DigitalHub, and starting it."""

from __future__ import annotations

import digitalhub as dh

from building.preprocessing.ctx import isis
from dhub import configs, credentials
from dhub.paths import Function


def submitted(stage: Function, ref: str, **parameters) -> int:
    """Register a version of one stage from a pushed commit, and run it.

    Args:
        stage: The stage to submit, naming its function, handler and resources.
        ref: The branch, tag, or commit the platform clones.
        **parameters: What the handler is called with on the platform.

    Returns:
        code: A process exit code, zero once the job is started.
    """
    platform = configs.load()
    asked = platform.resources[stage.name.lower()]
    project = dh.get_or_create_project(platform.project)
    # The job installs the clone's requirements.txt at start, so no image is built
    function = project.new_function(
        name=stage.registered,
        kind="python",
        python_version=platform.python_version,
        base_image=platform.base_image,
        code_src=f"git+{platform.repository}#{ref}",
        handler=stage.value,
    )

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
