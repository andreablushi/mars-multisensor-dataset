"""How the evaluation set is labelled, once its config has been read."""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.analysis.models.rule import Rule


@dataclass(frozen=True, slots=True)
class Settings:
    """The settled choices for labelling the tiles the selection kept.

    Attributes:
        core: The share of a texture feature's box a tile has to lie in, about
            its centre.
        rules: Every class, in the order the config names them.
    """

    core: float
    rules: tuple[Rule, ...]
