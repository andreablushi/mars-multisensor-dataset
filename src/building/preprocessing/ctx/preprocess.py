"""Calibrating a raw CTX scan with ISIS, and projecting it onto the tiles it serves."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import tifffile

from building.configs import ctx as configs
from building.preprocessing.common import cut, geometry
from building.preprocessing.common.models.samples import Samples
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


def read_observation(identifier: str) -> CtxObservation:
    """Calibrate one placed scan and sample where its lines fall on the ground.

    Args:
        identifier: The observation, its files already in the download cache.

    Returns:
        observation: The calibrated scan and its sampled ground points.

    Raises:
        RuntimeError: When an ISIS application fails.
    """
    files = configs.CACHE.files(identifier, identifier)
    placed = files[configs.CUBE_SUFFIX]
    calibrated, cube, points, table = (
        placed.with_suffix(suffix) for suffix in (".cal.cub", ".eo.cub", ".txt", ".csv")
    )
    run_isis("ctxcal", {"from": placed, "to": calibrated})
    placed.unlink()
    label = read_cube_label(calibrated)
    if int(label.get("SpatialSumming", 1)) > 1:
        calibrated.replace(cube)
    else:
        run_isis("ctxevenodd", {"from": calibrated, "to": cube})
        calibrated.unlink()
    lines, samples = int(label["Lines"]), int(label["Samples"])
    sampled = np.unique(np.append(np.arange(1, lines + 1, configs.LINE_STEP), lines))
    across = np.linspace(1, samples, configs.SAMPLES_ACROSS).round().astype(int)
    points.write_text("".join(f"{s},{at}\n" for at in sampled for s in across))
    run_isis(
        "campt",
        {
            "from": cube,
            "usecoordlist": "true",
            "coordlist": points,
            "coordtype": "image",
            "format": "flat",
            "to": table,
            "allowoutside": "true",
        },
    )
    with table.open() as held:
        rows = [row for row in csv.DictReader(held) if row["PlanetocentricLatitude"]]
    said = files[configs.METADATA_SUFFIX]
    return CtxObservation(
        identifier,
        json.loads(said.read_text()) if said.exists() else {},
        cube,
        lines,
        np.array([float(row["Line"]) for row in rows]),
        np.array([float(row["PlanetocentricLatitude"]) for row in rows]),
        np.array([float(row["PositiveEast360Longitude"]) for row in rows]),
    )


def windowed(image: Path, bounds: tuple[np.ndarray, ...]) -> np.ndarray:
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
    return geometry.taken(window, (lines - top, samples - left))


def crop(observation: CtxObservation, frame: Tile) -> CtxSample | None:
    """Return one scan projected by ISIS onto its tile, holding only what the box keeps.

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
    grid = frame.grid
    template.write_text(
        configs.MAP.format(name=projection.EQUATORIAL, latitude=0.0, longitude=180.0)
        if grid is None
        else configs.MAP.format(
            name=projection.POLAR,
            latitude=90.0 if grid[1] else -90.0,
            longitude=grid[0],
        )
    )
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
        down, across, polar = projection.grid_axes(label)
        held = cut.overlap(Samples(down, across, True, polar), frame)
        if held is None:
            return None
        pixels = windowed(image, held.bounds)
    finally:
        for path in work.parent.glob(f"{work.name}.*"):
            path.unlink()
    return CtxSample(
        identifier=observation.identifier,
        position=held.position,
        label=label,
        inside=held.inside,
        valid=geometry.marked(pixels != BLANK),
        image=pixels,
    )
