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
        suffixes: The suffixes downloaded per kind, or keyed by None for all kinds.
        subdirectories: The directory each kind is kept in, for those kept apart.
    """

    root: Path
    suffixes: dict[str | None, tuple[str, ...]]
    subdirectories: dict[str, str] = field(default_factory=dict)

    def files(
        self, directory: str, stem: str, kind: str | None = None
    ) -> dict[str, Path]:
        """Return where each half of one product belongs.

        Args:
            directory: The directory under the root, the observation or a shared name.
            stem: What each half of the product is called, without its suffix.
            kind: Which product it is, or None for a single-kind instrument.

        Returns:
            files: The path for each suffix, keyed by suffix.

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
            directory: The observation or sheet directory the product was kept in.
        """
        # Only its own directory, so what every observation shares is left alone.
        shutil.rmtree(self.root / directory, ignore_errors=True)
