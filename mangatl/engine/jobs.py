"""The engine's job queue.

Batch jobs (whole chapters) run one at a time in submission order; interactive jobs (an editor
or agent request on one page) start at once on their own thread. Both share the GPU through
the engine's slot, where an interactive job's model calls go ahead of the batch's, so a
request waits for at most one model call of the batch, never for its page. Every job carries a
CancellationManager its function checks between steps; a cancelled batch job is resumed by
submitting it again (pages already written are skipped, see translate.py --skip-existing).
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from mangatl.common.cancellation import CancellationManager
from mangatl.common.exceptions import CancellationError
from mangatl.engine.gpu import Priority

_ids = itertools.count(1)


@dataclass
class Job:
    fn: Callable[[CancellationManager], Any]
    priority: Priority
    label: str = ""
    id: int = field(default_factory=lambda: next(_ids))
    status: str = "queued"  # queued, running, done, failed, cancelled
    cancel_manager: CancellationManager = field(default_factory=CancellationManager)
    future: Future = field(default_factory=Future)
    submitted: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def cancel(self) -> None:
        self.cancel_manager.cancel()
        if self.status == "queued":
            self.status = "cancelled"

    def result(self, timeout: float | None = None) -> Any:
        return self.future.result(timeout)

    def info(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "priority": self.priority.name.lower(),
            "status": self.status,
            "submitted": self.submitted,
            "started": self.started,
            "finished": self.finished,
        }


class JobQueue:
    def __init__(self) -> None:
        self._batch: queue.Queue[Job | None] = queue.Queue()
        self._jobs: list[Job] = []
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def submit(
        self,
        fn: Callable[[CancellationManager], Any],
        priority: Priority = Priority.BATCH,
        label: str = "",
    ) -> Job:
        job = Job(fn, priority, label)
        with self._lock:
            self._jobs.append(job)
            if priority == Priority.BATCH:
                if self._worker is None:
                    self._worker = threading.Thread(target=self._run_batches, name="batch-jobs", daemon=True)
                    self._worker.start()
                self._batch.put(job)
            else:
                threading.Thread(target=self._run, args=(job,), name=f"job-{job.id}", daemon=True).start()
        return job

    def _run_batches(self) -> None:
        while (job := self._batch.get()) is not None:
            self._run(job)

    def _run(self, job: Job) -> None:
        if job.status == "cancelled" or not job.future.set_running_or_notify_cancel():
            job.future.cancel()
            return
        job.status, job.started = "running", time.time()
        try:
            result = job.fn(job.cancel_manager)
        except CancellationError as e:
            job.status = "cancelled"
            job.future.set_exception(e)
        except BaseException as e:
            job.status = "cancelled" if job.cancel_manager.is_cancelled() else "failed"
            job.future.set_exception(e)
        else:
            job.status = "cancelled" if job.cancel_manager.is_cancelled() else "done"
            job.future.set_result(result)
        finally:
            job.finished = time.time()

    def jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs)

    def close(self) -> None:
        """Cancel every job and stop the batch worker (running jobs stop at their next check)."""
        with self._lock:
            for job in self._jobs:
                if job.status in ("queued", "running"):
                    job.cancel()
            worker, self._worker = self._worker, None
        if worker is not None:
            self._batch.put(None)
            worker.join(timeout=30)
