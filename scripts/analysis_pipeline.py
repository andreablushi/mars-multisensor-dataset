#!/usr/bin/env python
"""The analysis pipeline: run here by default, or submitted with --dh."""

from __future__ import annotations

import os
import time
from collections import Counter

from dhub import store, submit
from digitalhub_runtime_python import handler
from rich.console import Console

from mars_multisensor_dataset.analysis import paths, planner, runner
from mars_multisensor_dataset.analysis.console import print_summary
from mars_multisensor_dataset.analysis.coverage import artifacts as coverage_artifacts
from mars_multisensor_dataset.analysis.ground_truth.lavaflow.candidates import (
    candidate_labels as lavaflow_candidates,
)
from mars_multisensor_dataset.analysis.ground_truth.lavaflow.labels import (
    write_lavaflow_labels,
)
from mars_multisensor_dataset.analysis.ground_truth.nomenclature import catalogue
from mars_multisensor_dataset.analysis.ground_truth.nomenclature.candidates import (
    candidate_labels,
)
from mars_multisensor_dataset.analysis.ground_truth.nomenclature.draw import (
    drawn_labels,
)
from mars_multisensor_dataset.analysis.ground_truth.nomenclature.models.label import (
    Label,
)
from mars_multisensor_dataset.analysis.ground_truth.swim import maps
from mars_multisensor_dataset.analysis.ground_truth.swim.labels import write_ice_labels
from mars_multisensor_dataset.analysis.ground_truth.swim.sample import ice_labels
from mars_multisensor_dataset.analysis.metadata.fetchers.ancillary import (
    write_distortions,
)
from mars_multisensor_dataset.analysis.selector import select
from mars_multisensor_dataset.analysis.selector.artifacts import read_selected_tiles
from mars_multisensor_dataset.analysis.selector.filters.admit import floor_km2
from mars_multisensor_dataset.analysis.stats.artifacts import write_stats
from mars_multisensor_dataset.analysis.stats.dataset import dataset_stats
from mars_multisensor_dataset.analysis.utils.tile_group import every_tile, tile_grid
from mars_multisensor_dataset.building.configs import sharad
from mars_multisensor_dataset.common.config import analysis_settings
from mars_multisensor_dataset.common.console import PLAIN_LOG_ENV

SELECTION_ARTIFACTS = ("selection", "stats", "labels", "swim")


def compute_coverage(force: bool = False, workers: int | None = None) -> int:
    """Download the ODE metadata still missing, measure it, and read its ancillary.

    Args:
        force: Whether to redo finished work rather than skip it.
        workers: How many jobs each half runs at once, or None for the config.

    Returns:
        code: A process exit code, non zero when either half had a failure.
    """
    settings = analysis_settings(workers)
    console = Console()
    started_at = time.monotonic()
    downloaded, measured = runner.pipeline_outcomes(settings, console, force)
    elapsed = time.monotonic() - started_at
    coverage_artifacts.write_index()
    print_summary(downloaded, measured, elapsed, planner.unmeasured_sources(), console)
    unread = write_distortions(settings, force)
    console.print(f"ancillary: {unread} tables left unread")
    failed = any(outcome.failed for outcome in [*downloaded, *measured])
    return 1 if failed or unread else 0


def compute_labels(force: bool) -> list[Label]:
    """Label every kept tile, draw the set held out of training, and write it.

    Args:
        force: Whether to fetch the feature catalogue again rather than read it.

    Returns:
        labels: Every labelled tile, the drawn ones marked so.
    """
    settings = analysis_settings().ground_truth.nomenclature
    verdicts = catalogue.read_verdicts()
    refused = catalogue.refused_tiles(verdicts)
    labels = drawn_labels(
        candidate_labels(
            read_selected_tiles(),
            catalogue.read_features(refresh=force),
            settings,
        ),
        settings,
        verdicts,
    )
    catalogue.write_labels(labels)
    drawable = Counter(
        labelled.label for labelled in labels if labelled.tile not in refused
    )
    drawn = Counter(labelled.label for labelled in labels if labelled.drawn)
    for name in settings.classes:
        print(f"labels: {drawn[name]} of {drawable[name]} {name} tiles drawn")
    return labels


def compute_ice_labels(force: bool) -> None:
    """Label every tile the SWIM maps cover with the ice under it, and write it.

    Args:
        force: Whether to fetch the SWIM maps again rather than read them.
    """
    settings = analysis_settings().ground_truth.swim
    maps.download_maps(force)
    labelled = ice_labels(
        {tile.name: tile for tile in every_tile(tile_grid())}, settings
    )
    write_ice_labels(labelled)
    shallow = sum(one.ice_1_5m for one in labelled)
    deep = sum(one.ice_5m for one in labelled)
    print(
        f"swim: {len(labelled):,} tiles, {shallow:,} icy at 1 to 5 m,"
        f" {deep:,} below 5 m"
    )


def compute_lavaflow_labels() -> None:
    """List every crop a lava flow reflector is looked for in, and write it."""
    settings = analysis_settings()
    radargrams = next(
        one for one in settings.instrument_sets if one.iid == sharad.LAYOUT.instrument
    )
    labels = lavaflow_candidates(
        every_tile(tile_grid()),
        radargrams,
        settings.ground_truth.lavaflow,
        floor_km2(settings.criteria, radargrams.iid),
    )
    write_lavaflow_labels(labels)
    for name, region in settings.ground_truth.lavaflow.regions.items():
        held = [one for one in labels if one.region == name]
        named = {one.track for one in held if one.named}
        print(
            f"lavaflow: {name}, {len(named)} of {len(region.tracks)} named tracks,"
            f" {sum(one.named for one in held)} named crops,"
            f" {sum(not one.named for one in held)} unnamed"
        )


def compute_selection(force: bool = False, workers: int | None = None) -> None:
    """Search every measured tile under the filter, label it, and read what both left.

    Args:
        force: Whether to fetch the feature catalogue and the SWIM maps again.
        workers: How many processes to run on at once, or None for the config.
    """
    workers = analysis_settings(workers).workers
    # Read off the selection just written, so they never stand for an old filter
    selection, measured = select.select_dataset(workers)
    kept = sum(1 for selected in selection if selected.tile.kept)
    print(f"selection: {kept:,} of {len(selection):,} tiles kept", flush=True)
    write_stats(dataset_stats(measured, selection))
    drawn = {labelled.tile for labelled in compute_labels(force) if labelled.drawn}
    compute_ice_labels(force)
    compute_lavaflow_labels()
    write_stats(
        dataset_stats(
            [tile_stats for tile_stats in measured if tile_stats.window.tile in drawn],
            selection,
        ),
        paths.EVALUATION_STATS_ROOT,
    )


@handler(outputs=["coverage", "metadata", *SELECTION_ARTIFACTS])
def run_pipeline(project, force: bool = False, workers: int | None = None):
    """Run every stage on DigitalHub and publish everything each one left on disk.

    Args:
        project: The DigitalHub project the folders are logged into.
        force: Whether to redo finished work rather than skip it.
        workers: How many jobs each stage runs at once, as the job was sized.

    Returns:
        coverage: The folder of the coverage events and summaries.
        metadata: The folder of the ODE records the coverage was measured from.
        selection: The folder of the tiles and observations kept.
        stats: The folder of what the filter left of the dataset.
        labels: The folder of every labelled tile.
        swim: The folder of the SWIM maps and every tile they label.

    Raises:
        RuntimeError: When the measuring stage reported a failure.
    """
    os.environ[PLAIN_LOG_ENV] = "1"
    print("measuring coverage", flush=True)
    failed = compute_coverage(force, workers)
    coverage, metadata = (
        store.upload_folder(project, name, store.ARTIFACTS[name])
        for name in ("coverage", "metadata")
    )
    # Report a failure only once uploaded, and never select from short coverage
    if failed:
        raise RuntimeError("the run had failures; the folders hold what finished")
    store.download_verdicts(project)
    store.download_folder(project, "selection", store.ARTIFACTS["selection"])
    compute_selection(force, workers)
    print("done", flush=True)
    return (
        coverage,
        metadata,
        *(
            store.upload_folder(project, name, store.ARTIFACTS[name])
            for name in SELECTION_ARTIFACTS
        ),
    )


@handler(outputs=list(SELECTION_ARTIFACTS))
def run_selection(project, force: bool = False, workers: int | None = None):
    """Select the dataset on DigitalHub under the filter, and publish what it leaves.

    Args:
        project: The DigitalHub project the folders are logged into.
        force: Whether to fetch the feature catalogue and the SWIM maps again.
        workers: How many processes to run on at once, as the job was sized.

    Returns:
        selection: The folder of the tiles and observations kept.
        stats: The folder of what the filter left of the dataset.
        labels: The folder of every labelled tile.
        swim: The folder of the SWIM maps and every tile they label.
    """
    os.environ[PLAIN_LOG_ENV] = "1"
    for name in ("coverage", "metadata", "selection"):
        store.download_folder(project, name, store.ARTIFACTS[name])
    store.download_verdicts(project)
    compute_selection(force, workers)
    print("done", flush=True)
    return tuple(
        store.upload_folder(project, name, store.ARTIFACTS[name])
        for name in SELECTION_ARTIFACTS
    )


def main() -> int:
    """Run the pipeline where it was asked for, over the stages it was asked for.

    Returns:
        code: A process exit code, non zero when a stage failed.
    """
    parsed = submit.script_parser(__doc__, "redo finished work rather than skip it")
    parsed.add_argument(
        "--only-stats",
        action="store_true",
        help="skip the download and the measurement, and select, read the stats "
        "and label alone",
    )
    parsed.add_argument(
        "--only-lavaflow",
        action="store_true",
        help="list the lava flow crops alone, here, from the coverage on disk",
    )
    arguments = parsed.parse_args()

    if arguments.only_lavaflow:
        compute_lavaflow_labels()
        return 0
    if arguments.dh:
        stage = "selection" if arguments.only_stats else "pipeline"
        return submit.submitted(stage, arguments.ref, force=arguments.force)
    failed = 0 if arguments.only_stats else compute_coverage(arguments.force)
    compute_selection(force=arguments.force)
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
