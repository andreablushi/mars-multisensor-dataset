"""Reading the window section of the analysis config, which the filter is written in."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from analysis import paths
from analysis.selector.models.filter import Filter

# What the analysis config calls the section the filter is written in.
SECTION = "window"


def load(path: Path = paths.CONFIG_PATH) -> Filter:
    """Read what the instruments are asked for before a feature earns a place.

    Args:
        path: The analysis config, whose window section has to carry every setting.

    Returns:
        criteria: The filter every search runs under.

    Raises:
        ValueError: When a setting is missing or is not what it has to be.
    """
    spec = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(SECTION)
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"{path.name} should hold a `{SECTION}` section, found {spec!r}"
        )
    constraints = spec.get("constraints")
    if (
        isinstance(constraints, str)
        or not isinstance(constraints, Sequence)
        or not constraints
    ):
        raise ValueError(f"{path.name} needs a list of `constraints` under `{SECTION}`")
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
