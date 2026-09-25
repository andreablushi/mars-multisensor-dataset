"""How one instrument's arrays are laid out, which every stage of a build reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Axis(StrEnum):
    """What an axis holds: ground is placed, the others are the instrument's own."""

    GROUND = "ground"
    WAVELENGTH = "wavelength"
    DELAY = "delay"


@dataclass(frozen=True, slots=True)
class Layout:
    """What an instrument's arrays hold, declared once and apart from any stage.

    Attributes:
        instrument: The instrument, as ODE names it.
        dims: What each axis of its arrays is called.
        axes: What each of those axes holds, in the same order.
        measurement: The array stored for the instrument, as the sample names it.
        beside: The other arrays of the sample stored, by name, with their axes.
        stored: The type the measurement is written as, or None to keep its own.
        band_centres_nm: The shared band centres in nm, or None without wavelength.
    """

    instrument: str
    dims: tuple[str, ...]
    axes: tuple[Axis, ...]
    measurement: str
    beside: dict[str, tuple[str, ...]] = field(default_factory=dict)
    stored: str | None = None
    band_centres_nm: tuple[float, ...] | None = None
