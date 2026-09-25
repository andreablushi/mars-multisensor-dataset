"""One instrument set's downloaded observations, read off disk."""

from __future__ import annotations

from itertools import chain
from pathlib import Path

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
        stop, scale = item.get("UTC_stop_time"), item.get("Map_scale")
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
                stop=parse_timestamp(stop) if stop else None,
                wkt=wkt,
                map_scale_m=float(scale) if scale else None,
                north_wkt=north,
                south_wkt=south,
            )
        )
    observations.sort(key=lambda observation: (observation.start, observation.pdsid))
    return ObservationSet(
        set_key=set_key, observations=observations, discarded=discarded
    )
