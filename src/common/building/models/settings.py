"""What one build was asked to do, once the config has been read."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    """The settled choices for a build, whichever dataset it builds.

    Attributes:
        name: What this build is called, the directory it is written in and the
            name it is published under, so one build never overwrites another.
        workers: How many products are built at once, one per core, which a job
            a platform sized itself is given rather than reads.
        downloads: How many downloads run at once, which wait on the archives.
    """

    name: str
    workers: int
    downloads: int

    @property
    def in_flight(self) -> int:
        """Return how many products may be in the build at once.

        Returns:
            held: Enough waiting to feed every builder while every download is
                still in flight.
        """
        return self.workers + self.downloads
