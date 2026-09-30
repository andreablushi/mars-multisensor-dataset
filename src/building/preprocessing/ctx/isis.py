"""USGS ISIS, which calibrates CTX: how a job installs it, and how it is run."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from building.configs import ctx as configs
from common.pds import labels

ROOT = "/shared/isis"

DATA = "/shared/isisdata"

PREFIX = "/shared/mamba"

VERSION = "10.0.0"

ENVS = [
    {"name": "ISISROOT", "value": ROOT},
    {"name": "ISISDATA", "value": DATA},
    {"name": "LANG", "value": "C.UTF-8"},
]

# Before sharded indexes, whose hundreds of requests the cluster proxy refuses with 429
MAMBA = "https://conda.anaconda.org/conda-forge/linux-64/micromamba-1.5.9-0.tar.bz2"

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

ATTEMPTS = 3

# Each step with the seconds one attempt may take, since a stalled download never ends
INSTRUCTIONS = [
    (
        'python3 -c "import io,ssl,tarfile,urllib.request; '
        "tls=ssl.create_default_context(); tls.verify_flags&=~ssl.VERIFY_X509_STRICT; "
        "tarfile.open(fileobj=io.BytesIO("
        f"urllib.request.urlopen('{MAMBA}',context=tls,timeout=60).read()),"
        f"mode='r:bz2').extract('bin/micromamba','{PREFIX}',filter='data')\"",
        300,
    ),
    (
        f"rm -rf {ROOT} && export MAMBA_ROOT_PREFIX={PREFIX} MAMBA_DOWNLOAD_THREADS=2 "
        "MAMBA_REMOTE_MAX_RETRIES=10 MAMBA_REMOTE_BACKOFF_FACTOR=5 && "
        f"{PREFIX}/bin/micromamba create -y -q -p {ROOT} -c conda-forge "
        f"-c usgs-astrogeology isis={VERSION}",
        2400,
    ),
    *(
        (
            f"PATH={ROOT}/bin:$PATH ISISROOT={ROOT} downloadIsisData {mission} {DATA} "
            f'--include="{{{",".join(held)}}}"',
            1800,
        )
        for mission, held in HELD.items()
    ),
]


def install_isis() -> None:
    """Install ISIS and the data CTX needs onto the job's own disk.

    Raises:
        RuntimeError: When a step fails or stalls on every attempt.
    """
    for instruction, seconds in INSTRUCTIONS:
        for _ in range(ATTEMPTS):
            # A session of its own, so a stalled step is killed with its children
            with subprocess.Popen(
                instruction, shell=True, start_new_session=True
            ) as step:
                try:
                    if step.wait(timeout=seconds) == 0:
                        break
                except subprocess.TimeoutExpired:
                    os.killpg(step.pid, signal.SIGKILL)
                    step.wait()
        else:
            raise RuntimeError(f"ISIS failed {ATTEMPTS} times at: {instruction}")


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


def export_image(cube: Path, image: Path) -> None:
    """Write one cube as a TIFF of 16-bit counts over the reflectance range.

    Args:
        cube: The ISIS cube to export.
        image: Where the TIFF is written.

    Raises:
        RuntimeError: When isis2std fails.
    """
    low, high = configs.REFLECTANCE_RANGE
    run_isis(
        "isis2std",
        {
            "from": cube,
            "to": image,
            "format": "tiff",
            "bittype": "u16bit",
            "stretch": "manual",
            "minimum": low,
            "maximum": high,
        },
    )


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
