"""Registering a version of the pipeline on DigitalHub, and starting it."""

from __future__ import annotations

import digitalhub as dh

from building.preprocessing.ctx import isis
from dhub import configs
from dhub.paths import Function


def submitted(stage: Function, ref: str, **parameters) -> int:
    """Register a version of one stage from a pushed commit, and run it.

    Args:
        stage: The stage to submit, naming its function, handler and resources.
        ref: The branch, tag, or commit the platform clones.
        **parameters: What the handler is called with on the platform.

    Returns:
        code: A process exit code, zero once the job is started.

    Raises:
        RuntimeError: When an earlier run of the stage still holds a running pod.
    """
    platform = configs.load_platform()
    asked = platform.resources[stage.name.lower()]
    project = dh.get_or_create_project(platform.project)
    # A stopped run can keep its pod alive, so only a deleted one is surely gone
    alive = [
        run.id
        for run in dh.list_runs(project=platform.project)
        if f"/{stage.registered}:" in str(run.spec.function)
        and any(
            pod.get("status", {}).get("phase") == "Running"
            for pod in run.status.to_dict().get("k8s", {}).get("pods", [])
        )
    ]
    if alive:
        raise RuntimeError(f"{stage.registered} still runs as {alive}; delete them.")
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
        profile=asked.profile + ("-shared" if asked.shared else ""),
        resources={"cpu": str(asked.cpu), "mem": asked.memory, "disk": asked.disk},
        secrets=["DHCORE_PERSONAL_ACCESS_TOKEN"],
        envs=[
            {"name": "PYTHONPATH", "value": f"{root}:{root}/src:{root}/scripts"},
            *(isis.ENVS if asked.isis else []),
        ],
        parameters=parameters | {"workers": asked.cpu},
        wait=False,
    )
    print(run.key)
    return 0
