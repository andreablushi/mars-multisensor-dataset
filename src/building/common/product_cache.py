"""The cache each instrument keeps its downloaded products in."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from building.common.naming import Naming


@dataclass(frozen=True, slots=True)
class ProductCache:
    """Where one instrument keeps every product it downloads.

    Attributes:
        root: The directory the instrument downloads under.
        naming: How the instrument spells an observation and its products.
        suffixes: The suffixes each kind is downloaded as, None serving every other.
        subdirectories: The subdirectory each kind kept apart from its observation uses.
    """

    root: Path
    naming: Naming
    suffixes: dict[str | None, tuple[str, ...]]
    subdirectories: dict[str, str] = field(default_factory=dict)

    def files(
        self, directory: str, stem: str, kind: str | None = None
    ) -> dict[str, Path]:
        """Return where each file of one product belongs, keyed by suffix.

        Args:
            directory: The directory under the root, the observation or a shared name.
            stem: The name of each file of the product, without its suffix.
            kind: Which product it is, or None for the suffixes keyed by None.

        Returns:
            files: The path of each file, keyed by its suffix.

        Raises:
            KeyError: When the kind is not one this instrument publishes.
        """
        place = self.root / directory / self.subdirectories.get(kind, "")
        wanted = self.suffixes[kind if kind in self.suffixes else None]
        return {suffix: place / f"{stem}{suffix}" for suffix in wanted}

    def product_files(
        self, identifier: str, kind: str, **written: str
    ) -> dict[str, Path]:
        """Return where each file of one product of an observation belongs.

        Args:
            identifier: The observation, which is also its directory.
            kind: Which product of it.
            written: Parts the identifier lacks, such as the CRISM detector.

        Returns:
            files: The path of each file, keyed by its suffix.
        """
        product = self.naming.product(identifier, kind, **written)
        return self.files(identifier, product, kind)

    def discard(self, directory: str) -> None:
        """Delete everything one product was downloaded as.

        Args:
            directory: The observation or sheet directory the product was kept in.
        """
        # Only its own directory, so what every observation shares is left alone.
        shutil.rmtree(self.root / directory, ignore_errors=True)
