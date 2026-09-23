"""A table published beside every product of a set, and the distortion of each look."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Ancillary:
    """What a set's ancillary product is, and which of its columns are read.

    Attributes:
        pt: The ODE product type the ancillary is published under.
        latitude: The column placing each row, in planetocentric degrees.
        longitude: The column placing each row, in degrees east.
        solar_zenith: The column holding each row's solar zenith angle, in degrees.
        night_above: The solar zenith angle past which a row is on the night side.
        distortion: The column holding each row's signal phase distortion.
    """

    pt: str
    latitude: str
    longitude: str
    solar_zenith: str
    night_above: float
    distortion: str


@dataclass(frozen=True, slots=True)
class Distortion:
    """The least signal phase distortion of one look's rows inside one group.

    Attributes:
        group: The tile group the look was listed in.
        pdsid: The PDS product identifier.
        night: The least of its rows on the night side, or None when none were.
        overall: The least of every row, or None when none fell inside.
    """

    group: str
    pdsid: str
    night: float | None
    overall: float | None
