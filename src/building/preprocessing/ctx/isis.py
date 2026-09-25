"""USGS ISIS, which calibrates CTX: how an image installs it, and how it is run."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from building.common.pds import labels

ROOT = "/opt/isis"

DATA = "/opt/isisdata"

VERSION = "10.0.0"

ENVS = [
    {"name": "ISISROOT", "value": ROOT},
    {"name": "ISISDATA", "value": DATA},
    {"name": "LANG", "value": "C.UTF-8"},
]

MAMBA = "https://micro.mamba.pm/api/micromamba/linux-64/latest"

HELD = {
    "mro": (
        "calibration/ctx*",
        "kernels/iak/**",
        "kernels/ik/**",
        "kernels/fk/**",
        "kernels/sclk/**",
    ),
    "base": (
        "kernels/lsk/**",
        "kernels/pck/**",
        "kernels/iak/**",
        "translations/**",
        "templates/**",
        "dems/molaMarsPlanetaryRadius0005*",
    ),
}

INSTRUCTIONS = [
    'python3 -c "import io,tarfile,urllib.request; tarfile.open(fileobj=io.BytesIO('
    f"urllib.request.urlopen('{MAMBA}').read()),mode='r:bz2')"
    ".extract('bin/micromamba','/usr/local')\"",
    f"export MAMBA_ROOT_PREFIX=/opt/mamba && micromamba create -y -q -p {ROOT} "
    f"-c conda-forge -c usgs-astrogeology isis={VERSION} && micromamba clean -a -y",
    *(
        f"PATH={ROOT}/bin:$PATH ISISROOT={ROOT} downloadIsisData {mission} {DATA} "
        f'--include="{{{",".join(held)}}}"'
        for mission, held in HELD.items()
    ),
]


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
        # An image's LC_ALL=C outranks LANG, and Qt warns on every run without UTF-8
        env={**os.environ, "LC_ALL": "C.UTF-8"},
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
