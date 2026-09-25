"""Downloading one SHARAD track from ODE, only the columns its tiles keep."""

from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np

from building.configs import sharad as configs
from building.download import archive
from building.preprocessing.sharad.crop import kept_columns
from common.fetch import ranges
from common.models.tile import Tile
from common.pds import labels, tables

# What ODE publishes SHARAD under.
ODE = {"ihid": "MRO", "iid": "SHARAD"}

# The ODE product types a radargram, its geometry and its clutter are published under.
PRODUCT_TYPES = {
    configs.Kind.OBSERVATION: "USRDRV2",
    configs.Kind.GEOMETRY: "USGEOMV2",
    configs.Kind.CLUTTER: "SHSIMU",
}


def column_spans(
    columns: np.ndarray, lines: int, samples: int, itemsize: int, start: int
) -> tuple[tuple[int, int], ...]:
    """Return the bytes some columns of a line by line image take up in its file.

    Args:
        columns: The sorted columns to keep, counted from zero.
        lines: How many lines the image holds.
        samples: How many columns each line holds.
        itemsize: How many bytes one value takes.
        start: The byte of the file the image starts at.

    Returns:
        spans: The first and past-the-last byte of each run of columns, line by line.
    """
    breaks = np.flatnonzero(np.diff(columns) != 1) + 1
    runs = [
        (int(run[0]), int(run[-1]) + 1) for run in np.split(columns, breaks) if run.size
    ]
    row = samples * itemsize
    return tuple(
        (start + line * row + first * itemsize, start + line * row + last * itemsize)
        for line in range(lines)
        for first, last in runs
    )


def download_sparse_image(
    path: Path,
    url: str,
    client: httpx.Client,
    spans: tuple[tuple[int, int], ...],
    size: int,
    origin: int = 0,
) -> None:
    """Download some byte ranges of one image into a sparse copy, unless it is on disk.

    Args:
        path: Where the copy belongs.
        url: Where the image is served from, or empty when ODE offers none.
        client: The client the ranges are asked over.
        spans: The first and past-the-last byte of each range to keep.
        size: How many bytes the copy holds.
        origin: Which byte of the served file the copy starts at.

    Raises:
        FileNotFoundError: When the copy is missing and has no URL.
    """
    if path.exists():
        return
    if not url:
        raise FileNotFoundError(f"No .img offered for {path.stem}.")
    ranges.patched(
        url, path, spans, size, archive.TIMEOUT, client=client, origin=origin
    )


def fetch(identifier: str, client: httpx.Client, frames: tuple[Tile, ...]) -> None:
    """Download one track's geometry, and only the columns its tiles keep of the rest.

    Args:
        identifier: The observation to fetch.
        client: The client every query and download goes over.
        frames: The tiles it is cut to, which settle the columns downloaded.

    Raises:
        FileNotFoundError: When ODE offers no download for a product.
        FetchError: When the archive will not serve the byte ranges asked.
    """
    products = {kind: configs.NAMING.product(identifier, kind) for kind in configs.Kind}
    files = {
        kind: configs.CACHE.product_files(identifier, kind) for kind in configs.Kind
    }
    if all(path.exists() for held in files.values() for path in held.values()):
        return
    placing = files[configs.Kind.GEOMETRY]
    archive.download_product(
        client,
        products[configs.Kind.GEOMETRY],
        placing,
        pt=PRODUCT_TYPES[configs.Kind.GEOMETRY],
        **ODE,
    )
    radargram = files[configs.Kind.OBSERVATION]
    offered = archive.product_urls(
        client,
        products[configs.Kind.OBSERVATION],
        pt=PRODUCT_TYPES[configs.Kind.OBSERVATION],
        **ODE,
    )
    archive.download_files({".lbl": radargram[".lbl"]}, offered, client=client)
    lines, samples, _, _, dtype = labels.image_layout(labels.load(radargram[".lbl"]))
    # Every other column is left a hole, which `crop` never reads.
    columns = kept_columns(tables.load_table(placing[".tab"])[0], frames)
    itemsize = np.dtype(dtype).itemsize
    download_sparse_image(
        radargram[".img"],
        offered.get(".img", ""),
        client,
        column_spans(columns, lines, samples, itemsize, 0),
        lines * samples * itemsize,
    )
    # The combined simulation holds several arrays of the radargram's size in turn.
    itemsize = np.dtype(configs.CLUTTER_TYPE).itemsize
    size = lines * samples * itemsize
    start = configs.CLUTTER_ARRAY * size
    clutter = archive.product_urls(
        client,
        products[configs.Kind.CLUTTER],
        pt=PRODUCT_TYPES[configs.Kind.CLUTTER],
        **ODE,
    )
    download_sparse_image(
        files[configs.Kind.CLUTTER][".img"],
        clutter.get(".img", ""),
        client,
        column_spans(columns, lines, samples, itemsize, start),
        size,
        origin=start,
    )
