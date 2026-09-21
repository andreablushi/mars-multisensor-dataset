"""What one evaluation class is read from, once its config has been read."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Rule:
    """The features one class is read from, and how a tile earns it.

    Attributes:
        descriptor: The IAU descriptor it is read from, or None for named features.
        names: The features it is read from, or empty where the descriptor is used.
        diameter_km: The diameters an object is kept for, or None for a texture.
        latitudes: The latitudes a texture is kept to, or None for anywhere.
    """

    descriptor: str | None = None
    names: list[str] = field(default_factory=list)
    diameter_km: list[float] | None = None
    latitudes: list[float] | None = None
