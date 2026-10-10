"""CTX over one tile, as raw camera counts and as the build calibrates and cuts it."""

from __future__ import annotations

import shutil
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import tifffile

from analysis.visualization import panels
from analysis.visualization.preprocessing import product
from building.configs import ctx as configs
from building.preprocessing.ctx.crop import reached_lines
from building.preprocessing.ctx.isis import run_isis
from common.models.tile import Tile

NAME = "CTX"


def delivered_counts(cube: Path, lines: tuple[int, int]) -> np.ndarray:
    """Return the raw counts of a placed scan's lines, stretched.

    Args:
        cube: The placed scan, before calibration.
        lines: The first and last line to read, counted from one.

    Returns:
        counts: Lines by samples in camera geometry, from 0 to 1.

    Raises:
        RuntimeError: When an ISIS application fails.
    """
    first, last = lines
    trimmed, image = (cube.with_suffix(suffix) for suffix in (".raw.cub", ".raw.tif"))
    try:
        run_isis(
            "crop",
            {"from": cube, "to": trimmed, "line": first, "nlines": last - first + 1},
        )
        run_isis(
            "isis2std",
            {"from": trimmed, "to": image, "format": "tiff", "bittype": "u16bit"},
        )
        counts = tifffile.imread(image).astype(float)
    finally:
        trimmed.unlink(missing_ok=True)
        image.unlink(missing_ok=True)
    return panels.stretched(np.where(counts > 0, counts, np.nan))


def plot(identifier: str, frame: Tile) -> widgets.Widget:
    """Show one CTX scan over a tile, as raw counts and built."""
    product.fetch_product(NAME, identifier, frame)
    placed = configs.CACHE.files(identifier, identifier)[configs.CUBE_SUFFIX]
    kept = placed.with_suffix(".placed.cub")
    shutil.copyfile(placed, kept)
    try:
        observation = product.product_observation(NAME, identifier)
        sample = product.tile_sample(NAME, observation, frame)
        delivered = delivered_counts(kept, reached_lines(observation, frame))
        observation.cube.unlink()
    finally:
        kept.replace(placed)
    built = np.where(sample.measured_ground, sample.image, np.nan)
    return panels.side_by_side(
        [("Raw counts", delivered), ("Built sample", panels.stretched(built))]
    )
