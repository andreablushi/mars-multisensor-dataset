"""Reading one CTX scan off disk, calibrated by ISIS and sampled where it looked."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence

import numpy as np

from building.configs import ctx as configs
from building.preprocessing.ctx.isis import read_cube_label, run_isis
from building.preprocessing.ctx.models.observation import CtxObservation
from building.preprocessing.ctx.normalize import scan_statistics


def read_observation(
    identifier: str, clip: Sequence[float], stride: int
) -> CtxObservation:
    """Calibrate one placed scan and sample where its lines fall on the ground.

    Args:
        identifier: The observation, its files already in the download cache.
        clip: The low and high percentiles the scan's counts are clipped at.
        stride: How many lines and samples apart its statistics are measured.

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
    sampled = np.unique(
        np.append(np.arange(1, lines + 1, configs.SAMPLED_LINE_STEP), lines)
    )
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
        json.loads(said.read_text()) if said.exists() else {},
        cube,
        lines,
        np.array([float(row["Line"]) for row in rows]),
        np.array([float(row["PlanetocentricLatitude"]) for row in rows]),
        np.array([float(row["PositiveEast360Longitude"]) for row in rows]),
        scan_statistics(cube, clip, stride),
    )
