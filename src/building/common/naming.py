"""Reading and writing the ids an archive publishes one observation under."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Naming:
    """How one archive spells an observation and the products published for it.

    Attributes:
        pattern: The regex matching a product id or an identifier, naming its parts.
        identity: The format string writing an identifier from those parts.
        marks: The parts a product id must carry to be wanted, empty in an identifier.
        template: The format string writing a product id from its parts, or None.
        fields: The parts each kind writes into the template, by kind.
    """

    pattern: re.Pattern[str]
    identity: str
    marks: tuple[str, ...] = ()
    template: str | None = None
    fields: dict[str, dict[str, str]] = field(default_factory=dict)

    def parts(self, name: str) -> dict[str, str] | None:
        """Split one id into the parts its pattern names, empty where one is absent.

        Args:
            name: A product id or an identifier.

        Returns:
            parts: The named parts, or None when the pattern does not match.
        """
        match = self.pattern.match(name)
        if not match:
            return None
        return {part: found or "" for part, found in match.groupdict().items()}

    def parse(self, product_id: str) -> str | None:
        """Return the identifier of the observation one product id belongs to.

        Args:
            product_id: The id to read, in whichever case its archive spells it.

        Returns:
            identifier: The observation, or None when the id does not match or
                lacks a mark.
        """
        parts = self.parts(product_id.lower())
        if parts is None or not all(parts.get(mark) for mark in self.marks):
            return None
        return self.identity.format(**parts)

    def product(self, identifier: str, kind: str, **written: str) -> str:
        """Return the id the archive publishes one product of an observation under.

        Args:
            identifier: The observation, as `parse` returns it.
            kind: Which product of it.
            written: Parts the identifier lacks, such as the CRISM detector.

        Returns:
            product: The product id the archive knows that product by.

        Raises:
            ValueError: When the identifier is not one this can read.
            KeyError: When the kind is not one this instrument publishes.
        """
        parts = self.parts(identifier)
        if parts is None:
            raise ValueError(f"{identifier} is not an id this can read.")
        return self.template.format(**{**parts, **self.fields[kind], **written})
