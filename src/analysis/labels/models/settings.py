"""How the evaluation set is labelled, once its config has been read."""

from __future__ import annotations

from dataclasses import dataclass

from analysis.labels.models.rule import Rule


@dataclass(frozen=True, slots=True)
class Settings:
    """The settled choices for labelling the tiles the selection kept.

    Attributes:
        core: The share of a texture feature's box a tile has to lie in, about
            its centre.
        rules: Every class, in the order the config names them.
        per_class: How many tiles the draw takes of every class, or None for as
            many as the scarcest class holds.
        seed: The number the draw is made with, so it is the same every run.
    """

    core: float
    rules: tuple[Rule, ...]
    per_class: int | None
    seed: int
