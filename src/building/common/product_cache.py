"""The cache each instrument keeps its downloaded products in."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProductCache:
    """Where one instrument keeps every product it downloads.

    Attributes:
        root: The directory the instrument downloads under.
        suffixes: The suffixes a product is downloaded as, keyed by the kind it
            is, or keyed by None where every kind is downloaded as the same set.
        subdirectories: The directory a kind is kept in under the product's own,
            keyed by kind, for the kinds not kept beside the rest.
    """

    root: Path
    suffixes: dict[str | None, tuple[str, ...]]
    subdirectories: dict[str, str] = field(default_factory=dict)

    def files(
        self, directory: str, stem: str, kind: str | None = None
    ) -> dict[str, Path]:
        """Return where each half of one product belongs.

        Args:
            directory: The directory under the root, which is the observation
                for a product of one, and a name of its own for what every
                observation shares.
            stem: What each half of the product is called, without its suffix.
            kind: Which product it is, for an instrument publishing more than
                one, or None where it publishes a single kind.

        Returns:
            The path for each suffix, keyed by suffix.

        Raises:
            KeyError: When the kind is not one this instrument publishes.
        """
        place = self.root / directory
        if kind in self.subdirectories:
            place = place / self.subdirectories[kind]
        wanted = self.suffixes[kind if kind in self.suffixes else None]
        return {suffix: place / f"{stem}{suffix}" for suffix in wanted}

    def discard(self, directory: str) -> None:
        """Delete everything one product was downloaded as.

        Args:
            directory: The directory under the root the product was kept in,
                which is the observation or tile it belongs to.

        Returns:
            None.
        """
        # Only the product's own directory, so what every observation shares
        # sits beside it under a name of its own and is left alone.
        shutil.rmtree(self.root / directory, ignore_errors=True)
