"""Cutting one CTX scan to the tiles it was kept for, projected by ISIS onto each."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tifffile

from building.configs import ctx as configs
from building.preprocessing.common import cut, geometry
from building.preprocessing.ctx import projection
from building.preprocessing.ctx.isis import read_cube_label, run_isis
from building.preprocessing.ctx.models.observation import CtxObservation
from building.preprocessing.ctx.models.sample import BLANK, CtxSample
from common.maths import geodesy
from common.maths.geodesy import TURN
from common.models.tile import Tile
from common.pds import labels

# Nothing here reads the export's no-data tag: what a scan left blank is `BLANK`
logging.getLogger("tifffile").setLevel(logging.ERROR)


def kept_pixels(image: Path, bounds: tuple[np.ndarray, ...]) -> np.ndarray:
    """Return the pixels one cut keeps, reading no more of the scan than holds them.

    Args:
        image: The scan's TIFF, read chunk by chunk.
        bounds: The lines to keep and then the samples, as the cut left them.

    Returns:
        pixels: The pixels of those lines and samples, as lines by samples.
    """
    lines, samples = bounds
    # A box over the meridian keeps two ends of a strip, so the window spans both.
    top, left = int(lines.min()), int(samples.min())
    window = tifffile.imread(
        image,
        selection=(
            slice(top, int(lines.max()) + 1),
            slice(left, int(samples.max()) + 1),
        ),
    )
    return geometry.kept_part(window, (lines - top, samples - left))


def crop(observation: CtxObservation, frame: Tile) -> CtxSample | None:
    """Return one scan holding only the pixels its tile's box keeps, projected by ISIS.

    Args:
        observation: The calibrated scan.
        frame: The local frame of the tile it was kept for.

    Returns:
        sample: The scan cut to that tile, or None where it reaches none of it.

    Raises:
        RuntimeError: When an ISIS application fails.
    """
    span = geodesy.longitude_span(frame.west_lon, frame.east_lon)
    inside = (
        (frame.min_lat <= observation.latitude)
        & (observation.latitude <= frame.max_lat)
        & ((observation.longitude - frame.west_lon) % TURN <= span)
    )
    reached = observation.line[inside]
    if not reached.size:
        return None
    first = max(1, int(reached.min()) - configs.LINE_STEP)
    last = min(observation.lines, int(reached.max()) + configs.LINE_STEP)
    work = observation.cube.parent / frame.name
    trimmed, template, projected, image = (
        work.with_suffix(suffix) for suffix in (".cut.cub", ".map", ".map.cub", ".tif")
    )
    template.write_text(projection.map_template(frame.grid))
    low, high = configs.REFLECTANCE_RANGE
    try:
        run_isis(
            "crop",
            {
                "from": observation.cube,
                "to": trimmed,
                "line": first,
                "nlines": last - first + 1,
            },
        )
        run_isis(
            "cam2map",
            {
                "from": trimmed,
                "to": projected,
                "map": template,
                "pixres": "mpp",
                "resolution": configs.PIXEL_RESOLUTION_M,
                "warpalgorithm": configs.WARP_ALGORITHM,
                "patchsize": configs.PATCH_SIZE,
                "defaultrange": "map",
                "minlat": frame.min_lat,
                "maxlat": frame.max_lat,
                "minlon": frame.west_lon,
                "maxlon": frame.west_lon + span,
            },
        )
        run_isis(
            "isis2std",
            {
                "from": projected,
                "to": image,
                "format": "tiff",
                "bittype": "u16bit",
                "stretch": "manual",
                "minimum": low,
                "maximum": high,
            },
        )
        label = labels.merge(read_cube_label(projected), observation.label)
        held = cut.overlap(projection.grid_samples(label), frame)
        if held is None:
            return None
        pixels = kept_pixels(image, held.bounds)
    finally:
        for path in work.parent.glob(f"{work.name}.*"):
            path.unlink()
    return CtxSample(
        identifier=observation.identifier,
        position=held.position,
        label=label,
        inside=held.inside,
        valid=geometry.partial_mask(pixels != BLANK),
        image=pixels,
    )
