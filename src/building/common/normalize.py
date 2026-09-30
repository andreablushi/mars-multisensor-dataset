"""Standardising a whole build by each instrument's dataset constants, once built."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from building import paths
from building.common.layout import Axis
from building.configs import mola as mola_configs
from building.dispatcher import INSTRUMENTS
from building.metadata import observation
from building.metadata.dataset import read_normalization
from building.metadata.index import read_observation_metadata
from building.metadata.observation import (
    ObservationMetadata,
    measured_mask,
    measured_statistics,
)
from building.models.settings import Settings
from building.preprocessing.common.store import MEASURED
from common.disk import parquet
from common.disk.files import atomic_path

UNSCALED = (mola_configs.LAYOUT.instrument,)


def normalize_dataset(
    settings: Settings,
    root: Path,
    budget: int,
    fetch: Callable[[str, Path, Sequence[str]], None] | None,
    checkpoint: Callable[[], None] | None,
) -> None:
    """Standardise every crop of a build not yet standardised, and record it so.

    Args:
        settings: The settled choices for the build, naming it and its reference.
        root: The directory the build is written in, its index already there.
        budget: How many bytes of crops are held on disk before they go up.
        fetch: What brings objects of a named build back to disk, or None locally.
        checkpoint: What publishes the build as it stands, or None locally.

    Raises:
        FileNotFoundError: When the reference build records no constants.
    """
    records = read_observation_metadata(root)
    # A crop still on disk goes first, so no checkpoint sends it up unscaled
    pending = sorted(
        (
            at
            for at, one in enumerate(records)
            if not one.normalized and one.instrument not in UNSCALED
        ),
        key=lambda at: not (root / records[at].path).exists(),
    )
    if not pending:
        return
    constants = read_normalization(root)
    if constants is None:
        constants = _reference_normalization(settings, fetch) or {
            name: dataset_constants(
                [records[at] for at in pending if records[at].instrument == name]
            )
            for name in {records[at].instrument for at in pending}
        }
        # Frozen before any crop changes, so a resumed build applies the same ones
        manifest = root / paths.DATASET_MANIFEST_NAME
        described = json.loads(manifest.read_text())
        manifest.write_text(
            json.dumps(described | {"normalization": constants}, indent=2)
        )
    print(f"normalizing {len(pending):,} crops", flush=True)
    with ProcessPoolExecutor(max_workers=settings.workers) as pool:
        for batch in _batches(pending, records, budget if checkpoint else math.inf):
            missing = [records[at].path for at in batch]
            missing = [one for one in missing if not (root / one).exists()]
            if fetch and missing:
                fetch(settings.name, root, missing)
            scaled = pool.map(
                partial(normalized_crop, root),
                [records[at] for at in batch],
                [constants[records[at].instrument] for at in batch],
            )
            for at, one in zip(batch, scaled, strict=True):
                records[at] = one
            parquet.write(
                records, observation.SCHEMA, root / paths.OBSERVATION_METADATA_NAME
            )
            if checkpoint:
                checkpoint()


def _reference_normalization(
    settings: Settings, fetch: Callable[[str, Path, Sequence[str]], None] | None
) -> dict[str, dict[str, Any]] | None:
    """Return the constants of the build this one is standardised like, if any.

    Args:
        settings: The settled choices for the build, naming its reference.
        fetch: What brings objects of a named build back to disk, or None locally.

    Returns:
        normalization: The reference's constants, or None where it pools its own.

    Raises:
        FileNotFoundError: When the reference build records no constants.
    """
    if settings.reference is None:
        return None
    reference = paths.dataset_root(settings.reference)
    if fetch:
        fetch(settings.reference, reference, [paths.DATASET_MANIFEST_NAME])
    constants = read_normalization(reference)
    if constants is None:
        raise FileNotFoundError(f"{settings.reference} records no normalization.")
    return constants


def _batches(
    pending: Sequence[int], records: Sequence[ObservationMetadata], budget: float
) -> list[list[int]]:
    """Return the pending rows split so each run of them fits the disk.

    Args:
        pending: Which rows still wait, in the order they are done.
        records: Every row of the index.
        budget: How many bytes one run may hold, as float32 values.

    Returns:
        batches: The rows, each run under the budget but never empty.
    """
    batches: list[list[int]] = [[]]
    held = 0
    for at in pending:
        size = 4 * math.prod(records[at].shape)
        if batches[-1] and held + size > budget:
            batches.append([])
            held = 0
        batches[-1].append(at)
        held += size
    return batches


def dataset_constants(records: Sequence[ObservationMetadata]) -> dict[str, Any]:
    """Return the mean and std of every value an instrument's rows measured.

    Args:
        records: The rows of one instrument, their statistics not yet standardised.

    Returns:
        constants: The mean and std, per band where the rows carry bands, pooled
            exactly from each row's count, mean and std.
    """
    if records[0].band_valid_count is not None:
        counts = np.array([one.band_valid_count for one in records], np.float64)
        means, stds = (
            np.array(
                [[value or 0.0 for value in getattr(one, name)] for one in records],
                np.float64,
            )
            for name in ("band_mean", "band_std")
        )
    else:
        counts, means, stds = (
            np.array([getattr(one, name) for one in records], np.float64)
            for name in ("valid_count", "value_mean", "value_std")
        )
    total = counts.sum(axis=0)
    measured = total > 0
    # A band no row measured is never read, so it keeps the identity
    mean = np.divide(
        (counts * means).sum(axis=0), total, out=np.zeros_like(total), where=measured
    )
    square = np.divide(
        (counts * (stds**2 + means**2)).sum(axis=0),
        total,
        out=np.ones_like(total),
        where=measured,
    )
    std = np.where(measured, np.sqrt(square - mean**2), 1.0)
    return {"mean": mean.tolist(), "std": std.tolist()}


def normalized_crop(
    root: Path, record: ObservationMetadata, constants: dict[str, Any]
) -> ObservationMetadata:
    """Rewrite one crop standardised by its instrument's constants.

    Args:
        root: The directory the build is written in.
        record: The crop's row, not yet standardised.
        constants: The mean and std of its instrument, per band where it has bands.

    Returns:
        record: The row, its statistics those of the standardised values.
    """
    layout = INSTRUMENTS[record.instrument].layout
    path = root / record.path
    with np.load(path) as held:
        arrays = dict(held)
    values = arrays[layout.measurement]
    bands = record.band_valid_count
    measured = measured_mask(
        arrays[MEASURED],
        None if bands is None else np.asarray(bands) > 0,
        layout,
        values,
    )
    mean, std = (np.asarray(constants[key], np.float32) for key in ("mean", "std"))
    # Per band constants run along the wavelength axis
    along = [-1 if holds == Axis.WAVELENGTH else 1 for holds in layout.axes]
    scaled = np.where(
        measured,
        (values.astype(np.float32) - mean.reshape(along)) / std.reshape(along),
        0.0,
    ).astype(values.dtype)
    arrays[layout.measurement] = scaled
    with atomic_path(path) as tmp, tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    # Half floats overflow a sum of squares, so the stored values are measured wider
    described = measured_statistics(scaled.astype(np.float32), measured, layout)
    return replace(record, normalized=True, **described)
