"""Reading the window filter config, the one file the filter itself is written in."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

import shared.disk.paths as paths
from analysis.selector.models.filter import Filter


def load(path: Path = paths.FILTER_CONFIG_PATH) -> Filter:
    """Read what the instruments are asked for before a feature earns a place.

    Args:
        path: The filter config, which has to exist and carry every setting.

    Returns:
        criteria: The filter every search runs under.

    Raises:
        ValueError: When a setting is missing or is not what it has to be.
    """
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping):
        raise ValueError(f"{path.name} should hold its settings, found {spec!r}")
    constraints = spec.get("constraints")
    if (
        isinstance(constraints, str)
        or not isinstance(constraints, Sequence)
        or not constraints
    ):
        raise ValueError(f"{path.name} needs a list of `constraints`")
    for constraint in constraints:
        if not isinstance(constraint, Mapping) or not constraint:
            raise ValueError(
                f"{path.name} wants each constraint as instrument to share, "
                f"found {constraint!r}"
            )
    timeless = spec.get("timeless") or []
    if isinstance(timeless, str) or not isinstance(timeless, Sequence):
        raise ValueError(f"{path.name} wants `timeless` as a list")
    admits = spec.get("admits") or {}
    if not isinstance(admits, Mapping):
        raise ValueError(f"{path.name} wants `admits` as instrument to pixels")
    return Filter(
        constraints=tuple(
            {str(iid): float(share) for iid, share in constraint.items()}
            for constraint in constraints
        ),
        admits={str(iid): float(pixels) for iid, pixels in admits.items()},
        span_ls=float(spec["span_ls"]),
        timeless=frozenset(str(iid) for iid in timeless),
    )


FILTER: Filter = load()
