"""Putting one instrument set's stored footprints onto the ground of its feature."""

from __future__ import annotations

import numpy as np
from shapely import from_wkt

from analysis.coverage.models.observation import ProjectedObservation, ProjectedSet
from analysis.coverage.projection.geometry import footprints, sizing
from analysis.models.observation import ObservationSet


def project(loaded: ObservationSet) -> ProjectedSet:
    """Project one set's footprints onto its feature and cut them to it.

    Args:
        loaded: The set's stored observations, in chronological order.

    Returns:
        projected: The observations that landed on the feature, and the region they were
            cut to.
    """
    region = footprints.feature_region(loaded.feature)
    widths = sizing.track_widths(loaded.observations)
    shapes = footprints.projected_footprints(
        region,
        from_wkt(
            np.asarray(
                [observation.wkt for observation in loaded.observations], dtype=object
            )
        ),
        np.asarray([width or 0.0 for width in widths], dtype=float),
    )
    projected = []
    missed = 0
    for observation, width_m, shape in zip(
        loaded.observations, widths, shapes, strict=True
    ):
        if shape.is_empty:
            missed += 1
            continue
        width_km = width_m / 1000.0 if width_m is not None else None
        projected.append(
            ProjectedObservation(
                pdsid=observation.pdsid,
                ihid=observation.ihid,
                iid=observation.iid,
                pt=observation.pt,
                start=observation.start,
                stop=observation.stop,
                shape=shape,
                width_km=width_km,
                pixel_km2=sizing.ground_pixel_km2(
                    loaded.set_key, observation.map_scale_m, width_km
                ),
            )
        )
    return ProjectedSet(
        feature=loaded.feature,
        set_key=loaded.set_key,
        region=region,
        observations=projected,
        discarded=loaded.discarded + missed,
    )
