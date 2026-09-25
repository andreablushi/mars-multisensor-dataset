"""Handing a build's products to the cores as they land, the cheapest first."""

from __future__ import annotations

import heapq
import itertools
import queue
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import httpx

from building.dispatcher import INSTRUMENTS, Archive
from building.models.job import Job, Outcome
from building.models.progress import Progress, Stage
from building.models.settings import Settings


class Scheduler:
    """Where every product of one build is, between its archive and its core."""

    def __init__(
        self,
        ode: httpx.Client,
        fetching: dict[str, ThreadPoolExecutor],
        building: ProcessPoolExecutor,
        build: Callable[[Job, Path], Outcome],
        root: Path,
        settings: Settings,
        progress: Progress,
    ) -> None:
        """Open a schedule over the pools a build runs on.

        Args:
            ode: The client every download is asked through.
            fetching: The threads the downloads run on, by the archive they wait on.
            building: The processes the builds run on.
            build: What builds one downloaded product in a process.
            root: The dataset's own root directory.
            settings: The settled choices, bounding how many products run at once.
            progress: What every product still in the build is doing.
        """
        self._ode = ode
        self._fetching = fetching
        self._building = building
        self._build = build
        self._root = root
        self._progress = progress
        self._finished: queue.Queue[Outcome] = queue.Queue()
        # Places per archive, so one waiting on the cores never stalls another
        self._places = {
            archive: threading.Semaphore(settings.workers + downloads)
            for archive, downloads in settings.downloads.items()
        }
        self._ready: list[tuple[int, int, Job, object]] = []
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
        ticket = self._progress.entered(job.label, Stage.FETCHING)
        try:
            instrument.fetch(job.identifier, self._ode, job.frames)
            if instrument.place is None:
                self._lined_up(job, ticket)
                return
            # Placing waits on the SPICE server, so it never holds an archive's thread
            self._progress.moved(ticket, Stage.PLACING)
            self._fetching[Archive.SPICE].submit(self._placed, job, ticket)
        except Exception as error:  # noqa: BLE001
            self._finish(Outcome(job, error=error), ticket)

    def _placed(self, job: Job, ticket: object) -> None:
        """Place one fetched product and hand it on to be built.

        Args:
            job: The product to place.
            ticket: What the job is tracked by while it is still in the build.
        """
        try:
            INSTRUMENTS[job.instrument].place(job.identifier)
        except Exception as error:  # noqa: BLE001
            self._finish(Outcome(job, error=error), ticket)
            return
        self._lined_up(job, ticket)

    def _lined_up(self, job: Job, ticket: object) -> None:
        """Line one ready product up for a core, and start it if one is free.

        Args:
            job: The product to build.
            ticket: What the job is tracked by while it is still in the build.
        """
        self._progress.moved(ticket, Stage.WAITING)
        # The lightest build goes first, so a quick one never queues behind a CTX scan
        weight = INSTRUMENTS[job.instrument].worker_bytes
        with self._lock:
            heapq.heappush(self._ready, (weight, next(self._arrived), job, ticket))
        self._dispatch()

    def _dispatch(self) -> None:
        """Start as many waiting builds as there are free cores."""
        while True:
            with self._lock:
                if not self._cores or not self._ready:
                    return
                self._cores -= 1
                _, _, job, ticket = heapq.heappop(self._ready)
            self._progress.moved(ticket, Stage.BUILDING)
            try:
                started = self._building.submit(self._build, job, self._root)
            except Exception as error:  # noqa: BLE001
                # The pool is closing, so this builds nowhere.
                self._built(ticket, Outcome(job, error=error))
                continue
            started.add_done_callback(partial(self._done, job, ticket))

    def _done(self, job: Job, ticket: object, done: Future[Outcome]) -> None:
        """Record what one build left, a worker the pool lost included.

        Args:
            job: The job that was built.
            ticket: What the job was tracked by while it was still in the build.
            done: What the build pool left.
        """
        try:
            outcome = done.result()
        except Exception as error:  # noqa: BLE001
            outcome = Outcome(job, error=error)
        self._built(ticket, outcome)

    def _built(self, ticket: object, outcome: Outcome) -> None:
        """Give back the core one build held, and start the next.

        Args:
            ticket: What the job was tracked by while it was still in the build.
            outcome: What the build left, whether it was built or failed.
        """
        with self._lock:
            self._cores += 1
        self._finish(outcome, ticket)
        self._dispatch()

    def _finish(self, outcome: Outcome, ticket: object) -> None:
        """Record what one job left and give back its archive's place.

        Args:
            outcome: What the job left, whether it was built or failed.
            ticket: What the job was tracked by while it was still in the build.
        """
        self._progress.finish(ticket)
        self._finished.put(outcome)
        self._places[INSTRUMENTS[outcome.job.instrument].archive].release()
