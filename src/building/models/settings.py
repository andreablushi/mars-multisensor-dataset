"""What one build was asked to do, once the config has been read."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass(slots=True)
class BuildSettings:
    """The settled choices for a build, whichever dataset it builds.

    Attributes:
        name: The build's name, its directory and what it is published under.
        workers: How many products are built at once, one per core.
        downloads: How many downloads run at once from each archive, by its name.
        preprocessing: What each instrument's reader is handed, by its name.
        reference: The build whose constants this one is standardised by, or None to
            pool its own.
    """

    name: str
    workers: int
    downloads: dict[str, int]
    preprocessing: dict[str, dict[str, Any]]
    reference: str | None = None


@dataclass(slots=True)
class TrainingSettings(BuildSettings):
    """The settled choices for the training build, beside how every build runs.

    Attributes:
        share: The share of kept tiles to build, above zero up to one.
        seed: The draw's seed, so a smaller build is a reproducible subset.
    """

    share: float = MISSING
    seed: int = MISSING
