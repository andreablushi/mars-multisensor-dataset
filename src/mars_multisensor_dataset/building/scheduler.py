"""Handing a build's products to the cores, the cheapest first, and cutting each."""

from __future__ import annotations

import heapq
import itertools
import queue
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from mars_multisensor_dataset.building.metadata.observation import (
    ObservationMetadata,
    observation_metadata,
)
from mars_multisensor_dataset.building.models.instrument import INSTRUMENTS, Archive
from mars_multisensor_dataset.building.models.job import Job, Outcome
from mars_multisensor_dataset.building.models.progress import Progress, Stage
from mars_multisensor_dataset.building.models.settings import BuildSettings
from mars_multisensor_dataset.building.preprocessing.common import store
from mars_multisensor_dataset.common.pool import cancellable_pool


class Scheduler:
    """Where every product of one build is, between its archive and its core."""

    def __init__(
        self,
        ode: httpx.Client,
        fetching: dict[str, ThreadPoolExecutor],
        pools: ExitStack,
        root: Path,
        settings: BuildSettings,
        progress: Progress,
    ) -> None:
        """Open a schedule over the pools a build runs on.

        Args:
            ode: The client every download is asked through.
            fetching: The threads the downloads run on, by the archive they wait on.
            pools: What every build pool is opened on, and shut down with.
            root: The dataset's own root directory, which every build writes in.
            settings: The settled choices, bounding how many products run at once
                and handing each reader its own.
            progress: What every product still in the build is doing.
        """
        self._ode = ode
        self._fetching = fetching
        self._pools = pools
        self._workers = settings.workers
        self._building = self._building_pool()
        self._root = root
        self._preprocessing = settings.preprocessing
        self._progress = progress
        self._finished: queue.Queue[Outcome] = queue.Queue()
        # Places per archive, so one waiting on the cores never stalls another
        self._places = {
            archive: threading.Semaphore(settings.workers + downloads)
            for archive, downloads in settings.downloads.items()
        }
        self._ready: list[tuple[int, int, Job]] = []
        self._arrived = itertools.count()
        self._cores = settings.workers
        self._lock = threading.Lock()

    def _building_pool(self) -> ProcessPoolExecutor:
        """Return a fresh pool of build processes, shut down when the build ends."""
        return self._pools.enter_context(
            cancellable_pool(ProcessPoolExecutor(self._workers))
        )

    def outcomes(self, jobs: tuple[Job, ...]) -> Iterator[Outcome]:
        """Fetch every product and build it the moment a core is free, in any order.

        Args:
            jobs: The products to fetch and build, heaviest first.

        Yields:
            outcome: One outcome per job, in the order they finish.
        """
        # Every path leaves one outcome and gives its place back, or it waits for ever.
        for job in jobs:
            self._fetching[INSTRUMENTS[job.instrument].archive].submit(
                self._fetched, job
            )
        for _ in jobs:
            yield self._finished.get()

    def _fetched(self, job: Job) -> None:
        """Bring one product down and hand it on to be placed or built.

        Args:
            job: The product to fetch.
        """
        instrument = INSTRUMENTS[job.instrument]
        # The place is taken before the download, so the room is never given elsewhere.
        self._places[instrument.archive].acquire()
        self._progress.moved(job, Stage.FETCHING)
        try:
            instrument.fetch(job.identifier, self._ode, job.frames)
            if instrument.place is None:
                self._lined_up(job)
                return
            # Placing waits on the SPICE server, so it never holds an archive's thread
            self._progress.moved(job, Stage.PLACING)
            self._fetching[Archive.SPICE].submit(self._placed, job)
        except Exception as error:  # noqa: BLE001
            self._finish(Outcome(job, error=error))

    def _placed(self, job: Job) -> None:
        """Place one fetched product and hand it on to be built.

        Args:
            job: The product to place.
        """
        try:
            INSTRUMENTS[job.instrument].place(job.identifier)
        except Exception as error:  # noqa: BLE001
            self._finish(Outcome(job, error=error))
        else:
            self._lined_up(job)

    def _lined_up(self, job: Job) -> None:
        """Line one ready product up for a core, and start it if one is free.

        Args:
            job: The product to build.
        """
        self._progress.moved(job, Stage.WAITING)
        # The lightest build goes first, so a quick one never queues behind a CTX scan
        weight = INSTRUMENTS[job.instrument].worker_bytes
        with self._lock:
            heapq.heappush(self._ready, (weight, next(self._arrived), job))
        self._dispatch()

    def _dispatch(self) -> None:
        """Start as many waiting builds as there are free cores."""
        while True:
            with self._lock:
                if not self._cores or not self._ready:
                    return
                self._cores -= 1
                _, _, job = heapq.heappop(self._ready)
                pool = self._building
            self._progress.moved(job, Stage.BUILDING)
            try:
                started = pool.submit(
                    build_product, job, self._root, self._preprocessing
                )
            except Exception as error:  # noqa: BLE001
                # The pool is closing, so this builds nowhere.
                started = Future()
                started.set_exception(error)
            started.add_done_callback(partial(self._built, job, pool))

    def _built(
        self, job: Job, pool: ProcessPoolExecutor, done: Future[Outcome]
    ) -> None:
        """Record what one build left, give its core back, and start the next.

        Args:
            job: The job that was built.
            pool: The pool it was handed to.
            done: What the build pool left, a worker it lost included.
        """
        try:
            outcome = done.result()
        except Exception as error:  # noqa: BLE001
            outcome = Outcome(job, error=error)
        with self._lock:
            self._cores += 1
            if isinstance(outcome.error, BrokenProcessPool) and pool is self._building:
                self._building = self._building_pool()
        self._finish(outcome)
        self._dispatch()

    def _finish(self, outcome: Outcome) -> None:
        """Record what one job left, delete a failed one's download, and free its place.

        Args:
            outcome: What the job left, whether it was built or failed.
        """
        instrument = INSTRUMENTS[outcome.job.instrument]
        if outcome.failed and instrument.discard:
            instrument.discard(outcome.job.identifier)
        self._progress.finish(outcome.job)
        self._finished.put(outcome)
        self._places[instrument.archive].release()


def build_product(
    job: Job, root: Path, preprocessing: dict[str, dict[str, Any]]
) -> Outcome:
    """Cut one downloaded product to every tile that kept it, and write each.

    Args:
        job: The product to build, and the tiles to cut it to.
        root: The dataset's own root directory.
        preprocessing: What each instrument's reader is handed, by its name.

    Returns:
        outcome: The outcome, its written samples and the first error a cut raised.

    Raises:
        Exception: Whatever reading the product raised, collected as a failure.
    """
    instrument = INSTRUMENTS[job.instrument]
    written: list[ObservationMetadata] = []
    emptied: list[str] = []
    failed: Exception | None = None
    try:
        # Read once however many tiles want it, which is why the product is the unit.
        observation = instrument.read_observation(
            job.identifier, **preprocessing.get(job.instrument, {})
        )
        for frame in job.frames:
            try:
                sample = (
                    None if observation is None else instrument.crop(observation, frame)
                )
            except Exception as error:  # noqa: BLE001
                # A tile failing to cut is kept as the error, and the rest still cut.
                failed = failed or error
                continue
            if sample is None or not sample.measured:
                emptied.append(frame.name)
                continue
            path = store.write_sample(
                sample, instrument.layout, frame, job.identifier, root
            )
            written.append(
                observation_metadata(
                    sample,
                    frame,
                    instrument.layout,
                    job.identifier,
                    str(path),
                    t_start=job.t_start,
                )
            )
    finally:
        # A product goes once every tile that wanted it is cut; it is a cache.
        if instrument.discard:
            instrument.discard(job.identifier)
    return Outcome(job, records=tuple(written), emptied=tuple(emptied), error=failed)
