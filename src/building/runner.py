"""Running a build's work: the downloads it waits on, and the crops it computes."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing
from functools import partial
from pathlib import Path

import httpx
from rich.console import Console

import utils.disk.paths as paths
from building import console as printing
from building import planner
from building.dispatcher import INSTRUMENTS
from building.metadata import read as metadata_read
from building.metadata import record
from building.metadata import write as metadata
from building.metadata.models.observation import ObservationRecord
from building.models.job import Job, Outcome, Plan
from building.models.settings import Settings


def run_build(
    settings: Settings,
    console: Console,
    root: Path = paths.DATASET_ROOT,
    *,
    force: bool = False,
) -> list[Outcome]:
    """Fetch every product a build needs and cut each to the features that kept it.

    A download waits on the network and a crop waits on the processor, so the
    two run on pools of their own and a product is built the moment it lands.

    Args:
        settings: The settled choices for the build.
        console: The console to render on.
        root: The dataset's own root directory.
        force: Whether to rebuild crops that are already written.

    Returns:
        Every finished outcome, in completion order.

    Raises:
        FileNotFoundError: When no selection has been written to build from.
    """
    with httpx.Client() as ode:
        plan = planner.build_plan(settings, ode, root, force=force)
        printing.describe(plan, settings, console)
        if not plan.jobs:
            _indexed(plan, [], root)
            return []
        with (
            ProcessPoolExecutor(max_workers=settings.workers) as building,
            ThreadPoolExecutor(max_workers=settings.workers) as fetching,
        ):
            held = _outcomes(plan.jobs, ode, fetching, building, root, settings.ready)
            with closing(held) as outcomes:
                collected = printing.render(
                    outcomes, len(plan.jobs), "building", console
                )
    _indexed(plan, collected, root)
    return collected


def _outcomes(
    jobs: tuple[Job, ...],
    ode: httpx.Client,
    fetching: ThreadPoolExecutor,
    building: ProcessPoolExecutor,
    root: Path,
    ready: int,
) -> Iterator[Outcome]:
    """Fetch every product and build it the moment it lands, in whatever order.

    A download waits on the network and a build waits on the processor, so a
    product goes to the build pool as soon as it is on disk, however many of the
    downloads planned ahead of it are still running. The jobs are still handed to
    the pool heaviest first, so what changes is which of them is waited on and
    never which of them is started.

    Downloading is the faster of the two, so it is held to the room it was given:
    a product takes a place before it comes down and gives it back once it has
    been built, which is what stops the whole archive landing on disk before the
    first crop is written.

    Args:
        jobs: The products to fetch and build, heaviest first.
        ode: The client every download is asked through.
        fetching: The threads the downloads run on.
        building: The processes the builds run on.
        root: The dataset's own root directory.
        ready: How many downloaded products may wait at once to be built.

    Yields:
        One outcome per job, in the order they finish.
    """
    finished: queue.Queue[Outcome] = queue.Queue()
    waiting = threading.Semaphore(ready)

    def downloaded(job: Job) -> Outcome:
        """Take a place on disk for one product and bring it down into it.

        The place is taken before the download rather than after, so the room a
        product is about to need is never given away to another one, and it is
        given back once the product has been built.

        Args:
            job: The product to fetch.

        Returns:
            The outcome, holding nothing when the product came down and the
            error that stopped it otherwise.
        """
        waiting.acquire()
        try:
            INSTRUMENTS[job.instrument].fetch(job.identifier, ode)
            return Outcome(job)
        except Exception as error:  # noqa: BLE001
            return Outcome(job, error=error)

    def send_to_build(job: Job, done: Future[Outcome]) -> None:
        """Hand a product that came down whole to the build pool.

        Every path here leaves exactly one outcome on the queue and gives back
        the one place it took, since a job that left neither would be waited on
        for ever.

        Args:
            job: The job whose product came down.
            done: What the download left, which `downloaded` never raises from.

        Returns:
            None.
        """
        outcome = done.result()
        if not outcome.failed:
            try:
                building.submit(build_product, job, root).add_done_callback(
                    partial(collect, job)
                )
                return
            except RuntimeError as error:
                # The build pool is shutting down, so this job builds nowhere.
                outcome = Outcome(job, error=error)
        finished.put(outcome)
        waiting.release()

    def collect(job: Job, done: Future[Outcome]) -> None:
        """Put what one job's build left on the queue, and give its place back.

        Args:
            job: The job that was built.
            done: What the build pool left, an outcome or the error it raised,
                a worker it lost included.

        Returns:
            None.
        """
        try:
            finished.put(done.result())
        except Exception as error:  # noqa: BLE001
            finished.put(Outcome(job, error=error))
        finally:
            waiting.release()

    for job in jobs:
        fetching.submit(downloaded, job).add_done_callback(partial(send_to_build, job))
    for _ in jobs:
        yield finished.get()


def build_product(job: Job, root: Path = paths.DATASET_ROOT) -> Outcome:
    """Cut one downloaded product to every feature that kept it, and write each.

    Args:
        job: The product to build, and the features to cut it to.
        root: The dataset's own root directory.

    Returns:
        The outcome, holding the record of every sample written, and the error
        that stopped it where one did after some were already on disk.

    Raises:
        Exception: Whatever reading the product off disk raised, which the pool
            hands back for the runner to collect as this job's own failure.
    """
    steps = INSTRUMENTS[job.instrument]
    # Read and cleaned once however many features want it, which is what makes
    # the product rather than the feature the unit of work.
    observation = steps.read_observation(job.identifier)
    written: list[ObservationRecord] = []
    missed = 0
    for frame in job.frames:
        try:
            held = steps.crop(observation, frame)
        except Exception as error:  # noqa: BLE001
            # What is already on disk is handed back, so a later failure never
            # leaves a written sample out of the index.
            return Outcome(job, records=tuple(written), error=error)
        # A product reaching none of a feature is no failure: the coverage it
        # was kept for is a box overlap, and a crop can still come out empty.
        if held is None:
            missed += 1
            continue
        path = steps.write(held, frame, root)
        written.append(
            record.observation_record(
                held,
                frame,
                steps.layout,
                str(path.relative_to(root)),
                t_start=job.t_start,
                altitude=steps.altitude(held) if steps.altitude else None,
            )
        )
    return Outcome(job, records=tuple(written), missed=missed)


def _indexed(plan: Plan, collected: Sequence[Outcome], root: Path) -> None:
    """Write the index over every crop the dataset holds, not this run's alone.

    A build that skips what is already written would otherwise index only what
    it rewrote, so what an earlier run left is read back and carried forward.

    Args:
        plan: What the build set out to do, whose frames every feature is in.
        collected: What every job of this run left.
        root: The dataset's own root directory.

    Returns:
        None.
    """
    written = [held for one in collected for held in one.records]
    rewritten = {
        (held.feature_class, held.feature_name, held.instrument, held.identifier)
        for held in written
    }
    try:
        standing = metadata_read.read_observation_records(root)
    except FileNotFoundError:
        standing = []
    kept = [
        held
        for held in standing
        if (held.feature_class, held.feature_name, held.instrument, held.identifier)
        not in rewritten
        and (root / held.path).exists()
    ]
    metadata.write_metadata(plan.frames, kept + written, root)
