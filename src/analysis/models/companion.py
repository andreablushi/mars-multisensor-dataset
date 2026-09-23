"""A table published beside every product of a set, summarised into its records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Companion:
    """What a set's companion product is, and which of its columns its records carry.

    Attributes:
        pt: The ODE product type the companion is published under.
        latitude: The column placing each row, in planetocentric degrees.
        longitude: The column placing each row, in degrees east.
        columns: The label column each record field is the median of, by field.
    """

    pt: str
    latitude: str
    longitude: str
    columns: dict[str, str]
