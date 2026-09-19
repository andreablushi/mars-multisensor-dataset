"""What the training build draws, once its config has been read."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    """The settled choices for the training build.

    Attributes:
        name: What this build is called, the directory it is written in and the
            name it is published under, so one build never overwrites another.
        share: What share of the tiles the selection kept to build, from above
            zero to one.
        seed: The number every draw is made with, so a smaller build is a
            reproducible subset of the full one.
    """

    name: str
    share: float
    seed: int
