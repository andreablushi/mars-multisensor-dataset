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
        version: Which layout the arrays and the index are written in, raised
            when what is written stops being readable by whatever read the
            version before it.
        cores: How many cores the run was given, for a job a platform sized
            itself, and None to read the machine's own. How many builds and
            downloads run at once is worked out from this and from the memory
            the machine has free, so neither is a setting a run carries.
    """

    name: str
    share: float
    instruments: tuple[str, ...]
    seed: int
    version: int
    cores: int | None = None
