"""One instrument set's downloaded observations, read off disk."""

from __future__ import annotations

import functools
from itertools import chain
from pathlib import Path

from analysis import paths
from analysis.models.observation import Observation, ObservationSet
from common.disk.files import read_jsonl
from common.pds.tables import parse_timestamp


def load_observations(path: Path) -> ObservationSet:
    """Read the observations stored for one group and instrument set.

    Args:
        path: The JSONL file holding the set's observations.

    Returns:
        observations: The set as stored, in chronological order.
    """
    stored = read_jsonl(path)
    first = next(stored)
    set_key = first["instrument_set"]
    observations: list[Observation] = []
    discarded = 0
    for item in chain([first], stored):
        # A record with no footprint or no start time cannot be placed at all
        wkt, start = item.get("Footprint_C0_geometry"), item.get("UTC_start_time")
        if not wkt or not start:
            discarded += 1
            continue
        north, south = (
            None if not polar or polar.endswith("EMPTY") else polar
            for polar in (
                item.get("Footprint_NP_geometry"),
                item.get("Footprint_SP_geometry"),
            )
        )
        observations.append(
            Observation(
                pdsid=item["pdsid"],
                ihid=item["ihid"],
                iid=item["iid"],
                pt=item["pt"],
                start=parse_timestamp(start),
                wkt=wkt,
                north_wkt=north,
                south_wkt=south,
            )
        )
    observations.sort(key=lambda observation: (observation.start, observation.pdsid))
    return ObservationSet(
        set_key=set_key, observations=observations, discarded=discarded
    )


@functools.lru_cache(maxsize=1)
def read_incidences(group: str) -> dict[str, float]:
    """Read the incidence angle ODE published for every look of one group, cached.

    Args:
        group: The name of the tile group.

    Returns:
        incidences: The solar zenith angle of each look at its centre, by pdsid.
    """
    return {
        record["pdsid"]: float(record["Incidence_angle"])
        for path in (paths.METADATA_ROOT / group).glob("*.jsonl")
        for record in read_jsonl(path)
        if record.get("Incidence_angle")
    }
