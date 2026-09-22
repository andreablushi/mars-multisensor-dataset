"""What one build was asked to do, once the config has been read."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    """The settled choices for a build, whichever dataset it builds.

    Attributes:
        name: The build's name, its directory and what it is published under.
        workers: How many products are built at once, one per core.
        downloads: How many downloads run at once from each archive, by its name.
    """

    name: str
    workers: int
    downloads: dict[str, int]

    @property
    def in_flight(self) -> int:
        """Return how many products may be in the build at once.

        Returns:
            held: Enough waiting to feed every builder while downloads are in flight.
        """
        return self.workers + sum(self.downloads.values())


@dataclass(slots=True)
class TrainingSettings(Settings):
    """The settled choices for the training build, beside how every build runs.

    Attributes:
        share: The share of kept tiles to build, above zero up to one.
        seed: The draw's seed, so a smaller build is a reproducible subset.
    """

    share: float = 1.0
    seed: int = 0
