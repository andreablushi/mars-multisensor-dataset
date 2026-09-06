"""Reading and writing the ids an archive publishes one observation under."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Naming:
    """How one archive spells the products of one observation.

    Attributes:
        pattern: What matches a product id, and matches an identifier too where
            the parts telling the kinds apart are absent.
        identity: How an identifier is written from the parts the pattern names.
        marks: The parts a published product carries that an identifier on its
            own does not, so an id holding all of them names a product.
        template: How a product id is written from an identifier's parts, or
            None where every product carries the observation's own name.
        fields: What each kind writes for the parts that tell the kinds apart,
            keyed by kind.
    """

    pattern: re.Pattern[str]
    identity: str
    marks: tuple[str, ...] = ()
    template: str | None = None
    fields: dict[str, dict[str, str]] = field(default_factory=dict)

    def parts(self, name: str) -> dict[str, str] | None:
        """Return the parts one id is written from.

        Args:
            name: A product id or an identifier.

        Returns:
            The part the pattern names, keyed by name and empty where the id
            leaves it out, or None when the id is not one this can read.
        """
        match = self.pattern.match(name)
        if not match:
            return None
        return {name: found or "" for name, found in match.groupdict().items()}

    def parse(self, product_id: str) -> str | None:
        """Read which observation a product id names.

        Args:
            product_id: The id to read, in whichever case its archive spells it.

        Returns:
            The identifier of the observation it belongs to, or None when the
            id is not a product this instrument wants.
        """
        parts = self.parts(product_id.lower())
        if parts is None or not all(parts.get(mark) for mark in self.marks):
            return None
        return self.identity.format(**parts)

    def product(self, identifier: str, kind: str, **written: str) -> str:
        """Return the id one product of an observation is published under.

        Args:
            identifier: The observation, as `parse` spells it.
            kind: Which product of it.
            written: What else is written into the id, for an archive naming a
                part the observation itself does not carry.

        Returns:
            The product id the archive knows that product by.

        Raises:
            ValueError: When the identifier is not one this can read.
            KeyError: When the kind is not one this instrument publishes.
        """
        parts = self.parts(identifier)
        if parts is None:
            raise ValueError(f"{identifier} is not an id this can read.")
        return self.template.format(**{**parts, **self.fields[kind], **written})
