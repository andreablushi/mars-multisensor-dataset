"""What the training build draws, once its config has been read."""

from __future__ import annotations

from dataclasses import dataclass

from common.building.models.settings import Settings as Building


@dataclass(slots=True)
class Settings(Building):
    """The settled choices for the training build, beside how every build runs.

    Attributes:
        share: What share of the tiles the selection kept to build, from above
            zero to one.
        seed: The number every draw is made with, so a smaller build is a
            reproducible subset of the full one.
    """

    share: float = 1.0
    seed: int = 0
