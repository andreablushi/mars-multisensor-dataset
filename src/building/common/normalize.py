"""The constants each instrument of a build is standardised by."""

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
    """Record in the manifest the mean, std, min and max of each instrument.

    Args:
        settings: The settled choices for the build, naming it and its reference.
        root: The directory the build is written in, its index already there.
        fetch: What brings objects of a named build back to disk, or None locally.

    Raises:
        FileNotFoundError: When the reference build records no constants.
    """
    records = read_observation_metadata(root)
    constants = _reference_normalization(settings, fetch)
    if constants is None:
        constants = {
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


def dataset_constants(
    records: Sequence[ObservationMetadata], prefix: str = "value"
) -> dict[str, Any]:
    """Return the mean, std, min and max of every value an instrument's rows measured.

    Args:
        records: The rows of one instrument, their statistics not yet standardised.
        prefix: What the row fields a scalar is pooled from start with.

    Returns:
        constants: The mean, std, min and max, per band where the rows carry bands
            with every band's pooled under `global`, pooled exactly from each row's.
    """
    constants = pooled_constants(records, "valid_count", prefix)
    if records[0].band_valid_count is None:
        return constants
    return pooled_constants(records, "band_valid_count", "band") | {"global": constants}


def pooled_constants(
    records: Sequence[ObservationMetadata], count: str, prefix: str
) -> dict[str, Any]:
    """Return the mean, std, min and max pooled from each row's own.

    Args:
        records: The rows pooled.
        count: The field holding each row's count, a scalar or one per band.
        prefix: What the fields holding each row's mean, std, min and max start with.

    Returns:
        constants: Scalars, or one per band, where a band no row measured keeps
            the identity and None as its extremes.
    """
    counts, means, stds, lows, highs = (
        np.array([getattr(one, name) for one in records], np.float64)
        for name in (
            count,
            *(f"{prefix}_{one}" for one in ("mean", "std", "min", "max")),
        )
    )
    total = counts.sum(axis=0)
    measured = total > 0
    # A band no row measured is never read, so it keeps the identity
    mean = np.divide(
        (counts * np.nan_to_num(means)).sum(axis=0),
        total,
        out=np.zeros_like(total),
        where=measured,
    )
    square = np.divide(
        (counts * np.nan_to_num(stds**2 + means**2)).sum(axis=0),
        total,
        out=np.ones_like(total),
        where=measured,
    )
    std = np.where(measured, np.sqrt(square - mean**2), 1.0)
    held = (counts > 0) & ~np.isnan(lows)
    low = np.min(lows, axis=0, where=held, initial=np.inf)
    high = np.max(highs, axis=0, where=held, initial=-np.inf)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": finite_or_none(low),
        "max": finite_or_none(high),
    }


def finite_or_none(values: np.ndarray) -> float | None | list[float | None]:
    """Return values as plain floats, None where one is not finite."""
    held = [float(one) if np.isfinite(one) else None for one in np.atleast_1d(values)]
    return held if np.ndim(values) else held[0]
