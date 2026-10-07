"""The constants each instrument of a build is standardised by, pooled once built."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from building import paths
from building.metadata.dataset import read_manifest, read_normalization, write_manifest
from building.metadata.index import read_observation_metadata
from building.metadata.observation import ObservationMetadata
from building.models.settings import BuildSettings


def record_normalization(
    settings: BuildSettings,
    root: Path,
    fetch: Callable[[str, Path, Sequence[str]], None] | None,
) -> None:
    """Record in the manifest the mean and std every instrument is standardised by.

    Args:
        settings: The settled choices for the build, naming it and its reference.
        root: The directory the build is written in, its index already there.
        fetch: What brings objects of a named build back to disk, or None locally.

    Raises:
        FileNotFoundError: When the reference build records no constants.
    """
    records = read_observation_metadata(root)
    constants = _reference_normalization(settings, fetch) or {
        name: dataset_constants([one for one in records if one.instrument == name])
        for name in {one.instrument for one in records}
    }
    write_manifest(replace(read_manifest(root), normalization=constants), root)


def _reference_normalization(
    settings: BuildSettings, fetch: Callable[[str, Path, Sequence[str]], None] | None
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
