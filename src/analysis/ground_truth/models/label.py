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
        foreign: How many other features reach into it, of another class or an
            excluded descriptor, or of its own where it holds an object.
        offset: How far its object sits from its centre, or it from its
            feature's, 0 at the centre and 1 at the edge.
        drawn: Whether the balanced draw took it into the evaluation set.
    """

    tile: str
    label: str
    feature: str
    foreign: int
    offset: float
    drawn: bool = False
