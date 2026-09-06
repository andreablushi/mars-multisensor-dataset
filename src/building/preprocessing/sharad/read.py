"""Reading one SHARAD observation off disk and placing its traces."""

from __future__ import annotations

from building.common.pds import images, tables
from building.configs import sharad as configs
from building.preprocessing.sharad.models.observation import SharadObservation

# The field the geometry names each radargram column in, counted from one.
COLUMN_FIELD = "RADARGRAM COLUMN"


def read_observation(identifier: str) -> SharadObservation:
    """Read one radargram and join it to the geometry it was measured at.

    Args:
        identifier: The observation, whose files must already be in the cache
            that `download.fetch` puts them in.

    Returns:
        The observation holding only the traces the geometry places, in the
        order the radargram stores them.

    Raises:
        FileNotFoundError: When either product or its label is missing.
        KeyError: When a label names a sample type this cannot read.
        ValueError: When the geometry holds fewer rows than its label promises.
    """
    # The echoes themselves, then the places they were sounded at.
    power = images.load_plane(
        configs.CACHE.files(
            identifier,
            configs.NAMING.product(identifier, configs.OBSERVATION),
            configs.OBSERVATION,
        )[".img"]
    )[0]
    geometry = tables.load_table(
        configs.CACHE.files(
            identifier,
            configs.NAMING.product(identifier, configs.GEOMETRY),
            configs.GEOMETRY,
        )[".tab"]
    )[0]
    # The geometry counts columns from one, and the radargram from zero.
    traces = geometry[COLUMN_FIELD].astype("i8") - 1
    return SharadObservation(identifier, power[:, traces], geometry, traces)
