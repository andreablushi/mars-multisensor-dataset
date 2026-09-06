"""What one build was asked to do, once the config has been read."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """The settled choices for a build, read from one flat config file.

    Attributes:
        name: What this build of the dataset is called, which is the directory
            it is written in and the name it is published under, so one build
            never overwrites another.
        share: What share of the features the selection kept to build, from
            above zero to one, drawn evenly across their classes. A feature is
            built whole or not at all, with every observation the selection
            left it.
        instruments: Which instruments to build, as ODE names them.
        seed: The number every draw is made with, so a smaller build is a
            reproducible subset of the full one.
        workers: How many jobs to run at once.
        ready: How many downloaded products may wait at once for the builds to
            reach them, which is what keeps the downloads from racing ahead.
    """

    name: str
    share: float
    instruments: tuple[str, ...]
    seed: int
    workers: int
    ready: int
