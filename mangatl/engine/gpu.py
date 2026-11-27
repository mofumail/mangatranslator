"""The GPU slot: one model call on the device at a time, the most urgent caller first.

Every model call holds the slot (mangatl.engine.Engine.use), so CUDA work never runs from two
threads at once (concurrent kernels from several threads crashed the context: "illegal memory
access", which a long-lived process can't recover from) and a waiting interactive request goes
ahead of every waiting batch call: it waits for one model call, never for a whole page.
"""

from __future__ import annotations

import heapq
import itertools
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from enum import IntEnum


class Priority(IntEnum):
    """Lower runs first."""

    INTERACTIVE = 0  # an editor or agent request a person is waiting on
    BATCH = 10  # pages of a batch job


class GpuSlot:
    """A priority mutex, FIFO within a priority. Re-entrant: the thread holding it may take it
    again (one model's call made inside another's)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._owner: int | None = None
        self._depth = 0
        self._waiting: list[tuple[int, int]] = []  # heap of (priority, arrival)
        self._arrivals = itertools.count()

    @contextmanager
    def hold(self, priority: Priority) -> Iterator[None]:
        me = threading.get_ident()
        with self._cond:
            if self._owner == me:
                self._depth += 1
            else:
                ticket = (int(priority), next(self._arrivals))
                heapq.heappush(self._waiting, ticket)
                while self._owner is not None or self._waiting[0] != ticket:
                    self._cond.wait()
                heapq.heappop(self._waiting)
                self._owner, self._depth = me, 1
        try:
            yield
        finally:
            with self._cond:
                self._depth -= 1
                if self._depth == 0:
                    self._owner = None
                    self._cond.notify_all()

    def outermost(self) -> bool:
        """Whether the calling thread holds the slot once (not inside another hold)."""
        with self._cond:
            return self._owner == threading.get_ident() and self._depth == 1

    def waiting(self) -> int:
        with self._cond:
            return len(self._waiting)
