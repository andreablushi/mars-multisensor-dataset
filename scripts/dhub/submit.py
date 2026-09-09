"""Registering a version of the pipeline on DigitalHub, and starting it."""

from __future__ import annotations

import tomllib

import digitalhub as dh

from building.models import budget
from dhub import configs, credentials
from shared import paths

UNITS = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4}


def given_bytes(memory: str) -> int:
    """Return how many bytes the memory a box was asked for comes to.

    Args:
        memory: The memory as the platform config spells it, such as `32Gi`.

    Returns:
        held: That memory in bytes.
    """
    unit = memory[-2:]
    if unit in UNITS:
        return int(memory[:-2]) * UNITS[unit]
    return int(memory)


def submitted(stage: str, handler: str, ref: str, **parameters) -> int:
    """Register a version of one stage from a pushed commit, and run it.

    Args:
        stage: Which stage to submit, naming the function it is registered as
            and the resources it is given.
        handler: The dotted path the platform imports and calls.
        ref: The branch, tag, or commit the platform clones.
        **parameters: What the handler is called with on the platform.

    Returns:
        code: A process exit code, non zero when the image did not build.
    """
    platform = configs.load()
    # The image is built from the repo's own dependencies, so it cannot drift
    manifest = (paths.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    needs = tomllib.loads(manifest)["project"]["dependencies"] + platform.image_extras
    project = dh.get_or_create_project(platform.project)
    function = project.new_function(
        name=platform.functions[stage],
        kind="python",
        python_version=platform.python_version,
        code_src=f"git+{platform.repository}#{ref}",
        handler=handler,
        requirements=needs,
    )

    # Build the image first, since the job cannot install anything itself.
    built = function.run(action="build", wait=True)
    if built.status.state != "COMPLETED":
        print(f"the image did not build: {built.status.state}")
        return 1
    function.refresh()

    # Start the job, told where the clone lands and what the box holds
    asked = platform.resources[stage]
    root = platform.source_root
    run = function.run(
        action="job",
        resources={"cpu": asked["cpu"], "mem": asked["memory"], "disk": asked["disk"]},
        secrets=[credentials.TOKEN],
        envs=[
            {"name": "PYTHONPATH", "value": f"{root}:{root}/src:{root}/scripts"},
            *credentials.minting_envs(),
            # What the build plans against, which is under the box so it may misjudge
            {
                "name": budget.MEMORY_ENV,
                "value": str(given_bytes(asked.get("budget", asked["memory"]))),
            },
        ],
        parameters=parameters | {"workers": int(asked["cpu"])},
        wait=False,
    )
    print(run.key)
    return 0
