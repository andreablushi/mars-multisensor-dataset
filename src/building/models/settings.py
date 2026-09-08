"""What one build was asked to do, once the config has been read."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """The settled choices for a build, read from one flat config file.

    Attributes:
        name: What this build is called, the directory it is written in and the
            name it is published under, so one build never overwrites another.
        share: What share of the features the selection kept to build, from above
            zero to one, drawn evenly across their classes.
        seed: The number every draw is made with, so a smaller build is a
            reproducible subset of the full one.
        max_observations: The observations a feature may keep and still be built,
            one seen more often left out whole rather than built in part.
        workers: How many products are built at once, one per core, which a job
            a platform sized itself is given rather than reads.
        downloads: How many downloads run at once, which wait on the archives.
    """

    name: str
    share: float
    seed: int
    max_observations: int
    workers: int
    downloads: int
