"""Selecting the dataset: every measured feature searched, and what is kept."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

from analysis.coverage.artifacts import index
from analysis.metadata.loaders.features import load_features
from analysis.models.feature import Feature
from analysis.selector.artifacts import filter_config as filtering
from analysis.selector.artifacts import write
from analysis.selector.models.selection import (
    SelectedFeature,
    SelectedObservation,
    Selection,
)
from analysis.selector.models.survey import Study

# One feature, as the catalogue spells its class and its name
FeatureName = tuple[str, str]
# Called with how many features are searched and how many there are
Progress = Callable[[int, int], None]


def select_dataset(workers: int, progress: Progress | None = None) -> list[Selection]:
    """Search every measured feature under the filter, and write the selection out.

    Args:
        workers: How many processes to search on at once, as the run is configured.
        progress: Called with how many features are searched and how many there are.

    Returns:
        What the search left of each feature, in catalogue order, leaving out a
        feature holding no measured set on disk.
    """
    features = index.catalogued_features()
    # The catalogue is read here so the selection carries each feature's own
    # ground, and a later run never has to open the catalogue again.
    catalogued = {(one.feature_class, one.name): one for one in load_features()}
    found: list[Selection | None] = [None for _ in features]
    observations = index.catalogued_observations()
    busiest_first = sorted(
        range(len(features)), key=lambda at: -observations.get(features[at], 0)
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        searched = pool.map(
            _searched,
            [catalogued[features[at]] for at in busiest_first],
            chunksize=1,
        )
        for done, (at, one) in enumerate(zip(busiest_first, searched, strict=True), 1):
            found[at] = one
            if progress is not None:
                progress(done, len(features))
    picked = [one for one in found if one is not None]
    write.write_selection(picked)
    return picked


def selected(study: Study, feature: Feature) -> Selection:
    """Read one feature's search as the rows the selection is written from.

    Args:
        study: What the search found over it.
        feature: The feature as the catalogue holds it, whose ground the row
            carries so a later run reads it from the selection alone.

    Returns:
        Its own row, and a row for each observation it keeps.
    """
    survey, track = study.survey, study.track
    row = SelectedFeature(
        feature_class=study.feature_class,
        feature_name=study.feature_name,
        min_lat=feature.min_lat,
        max_lat=feature.max_lat,
        west_lon=feature.west_lon,
        east_lon=feature.east_lon,
        kept=survey is not None,
        area_km2=track.grid.area_km2 if track else 0.0,
        start=survey.start if survey else None,
        end=survey.end if survey else None,
        days=survey.days if survey else 0.0,
        geo_mean=survey.geo_mean if survey else 0.0,
        taken=len(survey.taken) if survey else 0,
    )
    if survey is None or track is None:
        return Selection(feature=row)
    standing = set(survey.standing)
    return Selection(
        feature=row,
        observations=[
            SelectedObservation(
                feature_class=study.feature_class,
                feature_name=study.feature_name,
                ihid=track.observations[at].ihid,
                iid=track.observations[at].iid,
                pt=track.observations[at].pt,
                pdsid=track.observations[at].pdsid,
                t_start=track.observations[at].t_start,
                standing=at in standing,
            )
            for at in survey.taken
        ],
    )


def _searched(feature: Feature) -> Selection | None:
    """Search one feature and read it as the rows it is written as.

    Args:
        feature: The feature as the catalogue holds it.

    Returns:
        Its rows, and None where it has no measured set on disk.
    """
    coverage = index.load_feature(feature.feature_class, feature.name)
    if not coverage:
        return None
    return selected(Study.over(coverage, filtering.FILTER), feature)
