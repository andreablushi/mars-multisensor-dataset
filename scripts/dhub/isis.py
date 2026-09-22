"""ISIS, which calibrates and projects CTX, as images install and jobs find it."""

from __future__ import annotations

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
