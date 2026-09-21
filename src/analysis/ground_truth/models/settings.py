"""How the evaluation set is labelled, once its config has been read."""

from __future__ import annotations

from dataclasses import dataclass

from analysis.ground_truth.models.rule import Rule


@dataclass(slots=True)
class Settings:
    """The settled choices for labelling the tiles the selection kept.

    Attributes:
        excluded: The IAU descriptors a labelled tile is best kept clear of.
        classes: What every class is read from, by the class a tile earns, in
            the order the config names them.
        per_class: How many tiles the draw takes of every class, or None for as
            many as the scarcest class holds.
        seed: The number the draw is made with, so it is the same every run.
    """

    excluded: list[str]
    classes: dict[str, Rule]
    per_class: int | None
    seed: int
