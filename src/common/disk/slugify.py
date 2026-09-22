"""Helpers shared across the pipeline."""

from __future__ import annotations

import re

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Convert a name into a lowercase, underscore separated slug.

    Args:
        text: The raw name, for example "Rovers and Landers".

    Returns:
        slug: A slug such as "rovers_and_landers", or "unnamed" if empty.
    """
    slug = _SLUG_RE.sub("_", text.strip().lower()).strip("_")
    return slug or "unnamed"
