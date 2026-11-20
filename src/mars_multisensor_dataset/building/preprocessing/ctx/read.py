"""Reading one CTX scan off disk, calibrated by ISIS and sampled where it looked."""

from __future__ import annotations

import csv

import numpy as np

from mars_multisensor_dataset.building.configs import ctx as configs
from mars_multisensor_dataset.building.models.settings import Preprocessing
from mars_multisensor_dataset.building.preprocessing.ctx.isis import (
    read_cube_label,
    run_isis,
)
from mars_multisensor_dataset.building.preprocessing.ctx.models.observation import (
    CtxObservation,
)
from mars_multisensor_dataset.building.preprocessing.ctx.normalize import (
    scan_statistics,
)
from mars_multisensor_dataset.common.disk.files import read_json


def read_observation(identifier: str, preprocessing: Preprocessing) -> CtxObservation:
    """Calibrate one placed scan and sample where its lines fall on the ground.

    Args:
        identifier: The observation, its files already in the download cache.
        preprocessing: The build's reader settings, of which CTX reads its own.

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
    return CtxObservation(
        read_json(files[configs.METADATA_SUFFIX]),
        cube,
        lines,
        np.array([float(row["Line"]) for row in rows]),
        np.array([float(row["PlanetocentricLatitude"]) for row in rows]),
        np.array([float(row["PositiveEast360Longitude"]) for row in rows]),
        scan_statistics(cube, preprocessing.CTX.clip, preprocessing.CTX.stride),
        preprocessing.CTX.pool,
    )
