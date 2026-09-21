"""One tile the evaluation set labels, and the feature it was labelled from."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Label:
    """One kept tile, and the class the feature catalogue gives it.

    Attributes:
        tile: The tile's name, such as "b123_c0456".
        label: The class it earned, such as "chaos".
        feature: The feature it earned it from, such as "Aram Chaos".
        foreign: How many features of another class or excluded descriptor touch it.
        offset: How far it sits from its centre, 0 at the centre and 1 at the edge.
        drawn: Whether the balanced draw took it into the evaluation set.
    """

    tile: str
    label: str
    feature: str
    foreign: int
    offset: float
    drawn: bool = False
