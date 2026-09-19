"""What one evaluation class is read from, once its config has been read."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Rule:
    """The features one class is read from, and how a tile earns it.

    Attributes:
        descriptor: The IAU descriptor every feature of it is named under, or
            None where it is read from the named features alone.
        names: The features it is read from, or empty where the descriptor
            names them all.
        diameter_km: The diameters an object is kept for, a tile having to hold
            one whole, or None for a texture, which a patch of it shows.
        latitudes: The latitudes a texture is kept to, or None for anywhere.
    """

    descriptor: str | None = None
    names: list[str] = field(default_factory=list)
    diameter_km: list[float] | None = None
    latitudes: list[float] | None = None
