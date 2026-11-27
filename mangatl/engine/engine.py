"""The engine: one per process. Owns the loaded models, the GPU slot, the translator's
llama-server and the job queue; the pipeline reaches models only through it.

    engine = Engine(models_dir="models")
    models = engine.models(Priority.INTERACTIVE)      # what a job's pipeline calls carry
    with models.use(Model.HAYAI) as ocr:              # the slot is held inside the block only
        texts = ocr.read(crops)

Models load on first use and stay loaded. Before a load (and before each call) the engine
checks the device's free memory, which counts what other processes hold (llama-server, a game),
and unloads idle models until the call fits: the lowest `keep` first, then the least recently
used. A model is reachable only inside `use`, so an unloaded model is really gone (no caller
still holds it) and is never unloaded while it runs.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mangatl.common.log import log_message
from mangatl.engine.gpu import GpuSlot, Priority
from mangatl.engine.models import SPECS, Fetcher, LoadContext, Model, Spec, path_of

MB = 1024 * 1024
RESERVE_MB = 512  # kept free beyond a call's estimate (fragmentation, CUDA context growth)


@dataclass
class _Loaded:
    obj: Any
    weights_mb: float
    work_mb: float
    last_used: float = field(default_factory=time.monotonic)
    users: int = 0


class _Device:
    """Memory figures of the torch device (the engine's tests substitute their own)."""

    def __init__(self, device):
        self.device = device
        self.gpu = device.type == "cuda"

    def free_mb(self) -> float:
        """Free for this process: unused device memory plus torch's reserved-but-unused cache."""
        import torch

        free, _ = torch.cuda.mem_get_info(self.device)
        unused_cache = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        return (free + unused_cache) / MB

    def total_mb(self) -> float:
        import torch

        return torch.cuda.mem_get_info(self.device)[1] / MB

    def allocated_mb(self) -> float:
        import torch

        return torch.cuda.memory_allocated(self.device) / MB

    def start_peak(self) -> float:
        import torch

        torch.cuda.reset_peak_memory_stats(self.device)
        return self.allocated_mb()

    def peak_mb(self) -> float:
        import torch

        return torch.cuda.max_memory_allocated(self.device) / MB

    def release(self) -> None:
        from mangatl.common.device import empty_cache

        empty_cache(self.device)


class Engine:
    def __init__(
        self,
        models_dir: Path | str = "models",
        *,
        device: str = "auto",
        hf_token: str | None = None,
        hf_cache: bool = False,
        offline: bool = False,
        reserve_mb: float = RESERVE_MB,
        budget_mb: float | None = None,
        specs: dict[Model, Spec] | None = None,
        memory: Any = None,
    ):
        """device: "auto" (the best available) or "cpu". budget_mb caps the VRAM this engine's
        models may hold (default: whatever is free). specs / memory: substitutes for tests."""
        self.models_dir = Path(models_dir).resolve()
        self.fetch = Fetcher(self.models_dir, hf_token, hf_cache, offline)
        self.specs = specs or SPECS
        if memory is None:
            import torch

            from mangatl.common.device import get_best_device, get_best_dtype

            torch_device = torch.device("cpu") if device == "cpu" else get_best_device()
            memory = _Device(torch_device)
            self.dtype = get_best_dtype(torch_device)
        else:
            self.dtype = None
        self.memory = memory
        self.device = memory.device
        self.reserve_mb = reserve_mb
        self.budget_mb = budget_mb
        self.slot = GpuSlot()
        self._lock = threading.RLock()  # the registry below (loads run under the slot)
        self._loaded: dict[Model, _Loaded] = {}
        self._measured: dict[Model, tuple[float, float]] = {}  # survives unloads
        self._warned: set[Model] = set()
        self.loads = 0  # counters for status / tests
        self.unloads = 0
        self._llm = None
        self._jobs = None
        log_message(f"Engine on {self.device}", always_print=True)

    @classmethod
    def from_settings(cls, config, **kwargs) -> Engine:
        rt = config.runtime
        return cls(
            rt.models_dir,
            device=rt.device,
            hf_token=rt.hf_token,
            hf_cache=rt.hf_cache,
            budget_mb=rt.vram_budget_mb,
            **kwargs,
        )

    # ------------------------------------------------------------------ model access
    def models(self, priority: Priority = Priority.BATCH) -> Models:
        return Models(self, priority)

    def path(self, model: Model) -> Path:
        return path_of(self.models_dir, model)

    @contextmanager
    def use(self, model: Model, priority: Priority = Priority.BATCH) -> Iterator[Any]:
        spec = self.specs[model]
        if not spec.gpu:
            entry = self._entry(model, spec)
            entry.last_used = time.monotonic()
            yield entry.obj
            return
        with self.slot.hold(priority):
            outermost = self.slot.outermost()
            if self.memory.gpu:
                self._fit(model, spec)
            entry = self._entry(model, spec)
            entry.users += 1
            base = self.memory.start_peak() if outermost and self.memory.gpu else None
            try:
                yield entry.obj
            except Exception as e:
                if _is_oom(e):  # give the retry (the page is retried once) the whole device
                    self._evict(model, need_mb=float("inf"))
                raise
            finally:
                entry.users -= 1
                entry.last_used = time.monotonic()
                if base is not None:  # a call's own peak (nested calls share theirs)
                    entry.work_mb = max(entry.work_mb, self.memory.peak_mb() - base)
                    self._measured[model] = (entry.weights_mb, entry.work_mb)

    def _estimate(self, model: Model, spec: Spec) -> tuple[float, float]:
        return self._measured.get(model, (spec.weights_mb, spec.work_mb))

    def _entry(self, model: Model, spec: Spec) -> _Loaded:
        with self._lock:
            entry = self._loaded.get(model)
            if entry is not None:
                return entry
            log_message(f"Loading {model.value}...")
            t0 = time.monotonic()
            before = self.memory.allocated_mb() if spec.gpu and self.memory.gpu else 0.0
            obj = spec.load(LoadContext(self.device, self.dtype, self.fetch))
            weights = self.memory.allocated_mb() - before if spec.gpu and self.memory.gpu else 0.0
            _, work = self._estimate(model, spec)
            entry = self._loaded[model] = _Loaded(obj, weights or spec.weights_mb, work)
            self._measured[model] = (entry.weights_mb, entry.work_mb)
            self.loads += 1
            log_message(
                f"Loaded {model.value} in {time.monotonic() - t0:.1f}s"
                + (f" ({weights:.0f} MB)" if weights else "")
            )
            return entry

    def _fit(self, model: Model, spec: Spec) -> None:
        """Make room for a call of `model`: its working memory (and its weights, when it isn't
        loaded) must fit in free memory, and its weights within the budget."""
        weights, work = self._estimate(model, spec)
        with self._lock:
            new = 0 if model in self._loaded else weights
            self._evict(model, need_mb=work + self.reserve_mb + new, budget_mb=new)

    def _evict(self, keep: Model, need_mb: float, budget_mb: float = 0) -> None:
        """Unload idle models until `need_mb` is free on the device and `budget_mb` more fits in
        the budget."""
        with self._lock:
            while self.memory.free_mb() < need_mb or self._over_budget(budget_mb):
                idle = [
                    (m, e)
                    for m, e in self._loaded.items()
                    if m != keep and e.users == 0 and self.specs[m].gpu
                ]
                if not idle:
                    if need_mb != float("inf") and keep not in self._warned:
                        self._warned.add(keep)
                        log_message(
                            f"VRAM tight: {self.memory.free_mb():.0f} MB free for {keep.value} "
                            f"(wants {need_mb:.0f} MB)",
                            always_print=True,
                        )
                    return
                victim, _ = min(idle, key=lambda me: (self.specs[me[0]].keep, me[1].last_used))
                self._unload(victim)

    def _over_budget(self, more_mb: float) -> bool:
        if self.budget_mb is None:
            return False
        held = sum(e.weights_mb for m, e in self._loaded.items() if self.specs[m].gpu)
        return held + more_mb > self.budget_mb

    def _unload(self, model: Model) -> None:
        entry = self._loaded.pop(model)
        del entry.obj
        self.unloads += 1
        self.memory.release()
        log_message(f"Unloaded {model.value} (VRAM)", always_print=True)

    def unload(self, model: Model | None = None) -> None:
        """Unload one idle model (None: every idle model)."""
        with self._lock:
            for m in [model] if model is not None else list(self._loaded):
                if m in self._loaded and self._loaded[m].users == 0:
                    self._unload(m)

    def status(self) -> dict:
        """What is loaded and how much memory it holds (for logs and the API)."""
        with self._lock:
            loaded = {
                m.value: {
                    "weights_mb": round(e.weights_mb),
                    "work_mb": round(e.work_mb),
                    "idle_s": round(time.monotonic() - e.last_used, 1),
                    "in_use": e.users > 0,
                }
                for m, e in self._loaded.items()
            }
        out: dict[str, Any] = {
            "device": str(self.device),
            "models": loaded,
            "loads": self.loads,
            "unloads": self.unloads,
        }
        if self.memory.gpu:
            out.update(
                free_mb=round(self.memory.free_mb()),
                total_mb=round(self.memory.total_mb()),
                allocated_mb=round(self.memory.allocated_mb()),
            )
        if self._llm is not None:
            out["llm"] = self._llm.status()
        out["waiting"] = self.slot.waiting()
        return out

    # ------------------------------------------------------------------ llama-server, jobs
    @property
    def llm(self):
        from mangatl.engine.llm import LlamaServer

        with self._lock:
            if self._llm is None:
                self._llm = LlamaServer()
            return self._llm

    @property
    def jobs(self):
        from mangatl.engine.jobs import JobQueue

        with self._lock:
            if self._jobs is None:
                self._jobs = JobQueue()
            return self._jobs

    def close(self) -> None:
        if self._jobs is not None:
            self._jobs.close()
        if self._llm is not None:
            self._llm.stop()
        self.unload()


def _is_oom(e: BaseException) -> bool:
    return type(e).__name__ in ("OutOfMemoryError", "OutOfMemoryError_") or "out of memory" in str(e).lower()


class Models:
    """An engine's models, called at one priority: what a job's pipeline calls carry."""

    def __init__(self, engine: Engine, priority: Priority):
        self.engine = engine
        self.priority = priority

    @property
    def device(self):
        return self.engine.device

    def use(self, model: Model):
        return self.engine.use(model, self.priority)

    def path(self, model: Model) -> Path:
        return self.engine.path(model)

    def call(self, model: Model, fn: Callable[[Any], Any]) -> Any:
        with self.use(model) as m:
            return fn(m)
