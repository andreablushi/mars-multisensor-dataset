"""Running USGS ISIS, which a job finds through ISISROOT."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from building.common.pds import labels


def run_isis(app: str, parameters: dict[str, object]) -> None:
    """Run one ISIS application beside its input, raising what it said on failure.

    Args:
        app: The application, as ISIS names it.
        parameters: Its parameters, keyed as ISIS spells them, its input under "from".

    Raises:
        KeyError: When ISISROOT is unset, so no ISIS is installed here.
        RuntimeError: When the application fails.
    """
    done = subprocess.run(
        [
            str(Path(os.environ["ISISROOT"]) / "bin" / app),
            *(f"{key}={value}" for key, value in parameters.items()),
        ],
        capture_output=True,
        text=True,
        cwd=Path(str(parameters["from"])).parent,
    )
    if done.returncode:
        raise RuntimeError(f"{app}: {(done.stderr or done.stdout).strip()}")


def read_cube_label(cube: Path) -> dict[str, str]:
    """Return what the label of one ISIS cube says, written out beside it.

    Args:
        cube: The cube to read the label of.

    Returns:
        label: Its keys and values, as `labels.load` reads them.

    Raises:
        RuntimeError: When catlab fails.
    """
    described = cube.with_suffix(".lbl")
    run_isis("catlab", {"from": cube, "to": described, "append": "false"})
    return labels.load(described)
