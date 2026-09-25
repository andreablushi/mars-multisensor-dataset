"""Handing a build's products to the cores, the cheapest first, and cutting each."""

from __future__ import annotations

import heapq
import itertools
import queue
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import httpx

from building.dispatcher import INSTRUMENTS, Archive
from building.metadata.observation import ObservationMetadata, observation_metadata
from building.models.job import Job, Outcome
from building.models.progress import Progress, Stage
from building.models.settings import Settings
from building.preprocessing.common import store


class Scheduler:
    """Where every product of one build is, between its archive and its core."""

    def __init__(
        self,
        ode: httpx.Client,
        fetching: dict[str, ThreadPoolExecutor],
        building: ProcessPoolExecutor,
        root: Path,
        settings: Settings,
        progress: Progress,
    ) -> None:
        """Open a schedule over the pools a build runs on.

        Args:
            ode: The client every download is asked through.
            fetching: The threads the downloads run on, by the archive they wait on.
            building: The processes the builds run on.
            root: The dataset's own root directory, which every build writes in.
            settings: The settled choices, bounding how many products run at once.
            progress: What every product still in the build is doing.
        """
        self._ode = ode
        self._fetching = fetching
        self._building = building
        self._root = root
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
            return
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
            self._progress.moved(job, Stage.BUILDING)
            try:
                started = self._building.submit(build_product, job, self._root)
            except Exception as error:  # noqa: BLE001
                # The pool is closing, so this builds nowhere.
                started = Future()
                started.set_exception(error)
            started.add_done_callback(partial(self._built, job))

    def _built(self, job: Job, done: Future[Outcome]) -> None:
        """Record what one build left, give its core back, and start the next.

        Args:
            job: The job that was built.
            done: What the build pool left, a worker it lost included.
        """
        try:
            outcome = done.result()
        except Exception as error:  # noqa: BLE001
            outcome = Outcome(job, error=error)
        with self._lock:
            self._cores += 1
        self._finish(outcome)
        self._dispatch()

    def _finish(self, outcome: Outcome) -> None:
        """Record what one job left and give back its archive's place.

        Args:
            outcome: What the job left, whether it was built or failed.
        """
        self._progress.finish(outcome.job)
        self._finished.put(outcome)
        self._places[INSTRUMENTS[outcome.job.instrument].archive].release()


def build_product(job: Job, root: Path) -> Outcome:
    """Cut one downloaded product to every tile that kept it, and write each.

    Args:
        job: The product to build, and the tiles to cut it to.
        root: The dataset's own root directory.

    Returns:
        outcome: The outcome, its written samples and the first error a cut raised.

    Raises:
        Exception: Whatever reading the product raised, collected as a failure.
    """
    instrument = INSTRUMENTS[job.instrument]
    written: list[ObservationMetadata] = []
    missed = 0
    failed: Exception | None = None
    try:
        # Read once however many tiles want it, which is why the product is the unit.
        observation = instrument.read_observation(job.identifier)
        for frame in job.frames:
            try:
                sample = instrument.crop(observation, frame)
            except Exception as error:  # noqa: BLE001
                # A tile failing to cut is kept as the error, and the rest still cut.
                failed = failed or error
                continue
            # Reaching none of a tile is no failure, coverage being a box overlap.
            if sample is None:
                missed += 1
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
    return Outcome(job, records=tuple(written), missed=missed, error=failed)
