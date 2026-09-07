"""Loading one instrument set's downloaded observations off disk."""

from __future__ import annotations

from itertools import chain
from pathlib import Path

import analysis.metadata.provenance as provenance
from analysis.models.observation import Observation, ObservationSet
from utils.disk.files import read_jsonl


def load_observations(path: Path) -> ObservationSet:
    """Read the observations stored for one feature and instrument set.

    Args:
        path: The JSONL file holding the set's observations.

    Returns:
        observations: The set as stored, in chronological order.
    """
    stored = read_jsonl(path)
    first = next(stored)
    box, set_key = provenance.feature_of(first), provenance.set_key_of(first)
    observations: list[Observation] = []
    discarded = 0
    for item in chain([first], stored):
        # A record with no footprint or no start time cannot be placed at all
        wkt, start = item.get("Footprint_C0_geometry"), item.get("UTC_start_time")
        if not wkt or not start:
            discarded += 1
            continue
        stop, scale = item.get("UTC_stop_time"), item.get("Map_scale")
        observations.append(
            Observation(
                pdsid=item["pdsid"],
                ihid=item["ihid"],
                iid=item["iid"],
                pt=item["pt"],
                start=provenance.as_utc(start),
                stop=provenance.as_utc(stop) if stop else None,
                wkt=wkt,
                map_scale_m=float(scale) if scale else None,
            )
        )
    observations.sort(key=lambda observation: (observation.start, observation.pdsid))
    return ObservationSet(
        feature=box, set_key=set_key, observations=observations, discarded=discarded
    )
