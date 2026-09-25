"""Bringing one SHARAD track down from ODE, only the columns its tiles keep."""

from __future__ import annotations

import httpx
import numpy as np

from building.configs import sharad as configs
from building.download import archive
from building.preprocessing.sharad.crop import kept_columns
from common.models.tile import Tile
from common.pds import labels, tables

# What ODE publishes SHARAD under.
ODE = {"ihid": "MRO", "iid": "SHARAD"}

# The ODE product types a radargram, its geometry and its clutter are published under.
TYPES = {
    configs.Kind.OBSERVATION: "USRDRV2",
    configs.Kind.GEOMETRY: "USGEOMV2",
    configs.Kind.CLUTTER: "SHSIMU",
}


def column_spans(
    columns: np.ndarray, lines: int, samples: int, itemsize: int
) -> tuple[tuple[int, int], ...]:
    """Return the bytes some columns of a line by line image take up.

    Args:
        columns: The sorted columns to keep, counted from zero.
        lines: How many lines the image holds.
        samples: How many columns each line holds.
        itemsize: How many bytes one value takes.

    Returns:
        spans: The first and past-the-last byte of each run of columns, line by line.
    """
    breaks = np.flatnonzero(np.diff(columns) != 1) + 1
    runs = [
        (int(run[0]), int(run[-1]) + 1) for run in np.split(columns, breaks) if run.size
    ]
    row = samples * itemsize
    return tuple(
        (line * row + first * itemsize, line * row + last * itemsize)
        for line in range(lines)
        for first, last in runs
    )


def fetch(observation_id: str, client: httpx.Client, frames: tuple[Tile, ...]) -> None:
    """Bring one track's geometry down, and what its tiles keep of the rest.

    Args:
        observation_id: The observation to fetch.
        client: The client whose connections every query is asked over.
        frames: The tiles it is cut to, which settle the columns brought down.

    Raises:
        FileNotFoundError: When ODE offers no download for a product.
        FetchError: When the archive will not serve the byte ranges asked.
    """
    products = {
        kind: configs.NAMING.product(observation_id, kind) for kind in configs.Kind
    }
    files = {
        kind: configs.CACHE.files(observation_id, products[kind], kind)
        for kind in configs.Kind
    }
    if all(path.exists() for held in files.values() for path in held.values()):
        return
    placing = files[configs.Kind.GEOMETRY]
    archive.download_product(
        client,
        products[configs.Kind.GEOMETRY],
        placing,
        pt=TYPES[configs.Kind.GEOMETRY],
        **ODE,
    )
    radargram = files[configs.Kind.OBSERVATION]
    offered = archive.product_urls(
        client,
        products[configs.Kind.OBSERVATION],
        pt=TYPES[configs.Kind.OBSERVATION],
        **ODE,
    )
    archive.download_files({".lbl": radargram[".lbl"]}, offered, client=client)
    lines, samples, _, _, stored = labels.image_layout(labels.load(radargram[".lbl"]))
    # Every other column is left a hole, which `crop` never reads.
    columns = kept_columns(tables.load_table(placing[".tab"])[0], frames)
    itemsize = np.dtype(stored).itemsize
    archive.download_files(
        {".img": radargram[".img"]},
        offered,
        client=client,
        spans=column_spans(columns, lines, samples, itemsize),
        size=lines * samples * itemsize,
    )
    # The combined simulation holds several arrays of the radargram's size in turn.
    itemsize = np.dtype(configs.CLUTTER_TYPE).itemsize
    size = lines * samples * itemsize
    start = configs.CLUTTER_ARRAY * size
    archive.download_files(
        files[configs.Kind.CLUTTER],
        archive.product_urls(
            client,
            products[configs.Kind.CLUTTER],
            pt=TYPES[configs.Kind.CLUTTER],
            **ODE,
        ),
        client=client,
        spans=tuple(
            (start + first, start + last)
            for first, last in column_spans(columns, lines, samples, itemsize)
        ),
        size=size,
        origin=start,
    )
