"""The engine without torch: GPU slot order, the VRAM budget, the job queue, local-first fetching."""

import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from mangatl.engine.engine import Engine
from mangatl.engine.gpu import GpuSlot, Priority
from mangatl.engine.jobs import JobQueue
from mangatl.engine.models import Fetcher, Model, Spec


# --------------------------------------------------------------------------- GPU slot
def test_waiting_interactive_calls_go_before_waiting_batch_calls():
    slot = GpuSlot()
    order: list[str] = []
    holding = threading.Event()
    release = threading.Event()

    def holder():
        with slot.hold(Priority.BATCH):
            holding.set()
            release.wait()

    def caller(name, priority):
        with slot.hold(priority):
            order.append(name)

    t0 = threading.Thread(target=holder)
    t0.start()
    holding.wait()
    threads = []
    for name, priority in [
        ("batch1", Priority.BATCH),
        ("batch2", Priority.BATCH),
        ("click", Priority.INTERACTIVE),
    ]:
        t = threading.Thread(target=caller, args=(name, priority))
        t.start()
        threads.append(t)
        while slot.waiting() < len(threads):
            time.sleep(0.001)
    release.set()
    for t in [t0, *threads]:
        t.join()
    assert order == ["click", "batch1", "batch2"]  # FIFO within a priority


def test_the_slot_is_reentrant():
    slot = GpuSlot()
    with slot.hold(Priority.BATCH):
        assert slot.outermost()
        with slot.hold(Priority.INTERACTIVE):
            assert not slot.outermost()
        assert slot.outermost()
    assert slot.waiting() == 0


# --------------------------------------------------------------------------- budget
class FakeDevice:
    """A 1000 MB GPU; fake models hold memory until they are garbage."""

    def __init__(self, total=1000.0, other=0.0):
        self.device = SimpleNamespace(type="cuda")
        self.gpu = True
        self.total = total
        self.other = other  # held by other processes (llama-server)
        self.allocated = 0.0
        self.peak = 0.0

    def free_mb(self):
        return self.total - self.other - self.allocated

    def total_mb(self):
        return self.total

    def allocated_mb(self):
        return self.allocated

    def start_peak(self):
        self.peak = self.allocated
        return self.allocated

    def peak_mb(self):
        return self.peak

    def release(self):
        pass


class FakeModel:
    def __init__(self, device: FakeDevice, mb: float):
        device.allocated += mb
        weakref.finalize(self, lambda: setattr(device, "allocated", device.allocated - mb))
        self.device, self.mb = device, mb

    def run(self, work_mb: float = 0):
        self.device.peak = max(self.device.peak, self.device.allocated + work_mb)


def _engine(device, sizes: dict[Model, tuple[int, int]], keep=None, **kwargs):
    keep = keep or {}
    specs = {
        m: Spec("x", lambda ctx, mb=w: FakeModel(device, mb), w, work, keep.get(m, 5))
        for m, (w, work) in sizes.items()
    }
    return Engine("models", specs=specs, memory=device, reserve_mb=0, **kwargs)


def test_models_load_once_and_stay_loaded():
    device = FakeDevice()
    engine = _engine(device, {Model.HAYAI: (300, 100)})
    for _ in range(3):
        with engine.use(Model.HAYAI) as m:
            m.run()
    assert engine.loads == 1 and engine.unloads == 0
    assert engine.status()["models"]["hayai"]["weights_mb"] == 300


def test_the_least_recently_used_idle_model_makes_room():
    device = FakeDevice()
    engine = _engine(device, {Model.HAYAI: (400, 100), Model.PANELS: (300, 100), Model.LAMA: (400, 100)})
    with engine.use(Model.HAYAI):
        pass
    with engine.use(Model.PANELS):
        pass
    with engine.use(Model.LAMA):  # needs 500: HAYAI (used longest ago) goes
        pass
    assert set(engine.status()["models"]) == {"panels", "lama"}
    assert device.allocated == 700  # really freed: nobody kept a reference


def test_lower_keep_goes_first_and_a_model_in_use_never_goes():
    device = FakeDevice()
    engine = _engine(
        device,
        {Model.HAYAI: (300, 0), Model.LAMA: (300, 0), Model.TEXT_SAM: (300, 0), Model.PANELS: (300, 0)},
        keep={Model.LAMA: 1},
    )
    with engine.use(Model.LAMA):
        pass
    with engine.use(Model.HAYAI):
        pass
    with engine.use(Model.TEXT_SAM) as sam:  # 900 held; loading PANELS needs 300 more
        with engine.use(Model.PANELS):
            pass
        assert sam is not None
    assert set(engine.status()["models"]) == {"hayai", "text_sam", "panels"}  # LAMA (keep 1) went


def test_memory_other_processes_take_counts():
    device = FakeDevice(other=600)  # llama-server
    engine = _engine(device, {Model.HAYAI: (300, 0), Model.PANELS: (300, 0)})
    with engine.use(Model.HAYAI):
        pass
    with engine.use(Model.PANELS):
        pass
    assert set(engine.status()["models"]) == {"panels"}


def test_a_budget_caps_what_the_models_hold():
    device = FakeDevice()
    engine = _engine(device, {Model.HAYAI: (300, 0), Model.PANELS: (300, 0)}, budget_mb=500)
    with engine.use(Model.HAYAI):
        pass
    with engine.use(Model.PANELS):
        pass
    assert set(engine.status()["models"]) == {"panels"}


def test_a_budget_counts_weights_not_working_memory():
    device = FakeDevice(total=4000)
    engine = _engine(device, {Model.HAYAI: (300, 0), Model.LAMA: (300, 2000)}, budget_mb=700)
    for _ in range(2):
        with engine.use(Model.HAYAI):
            pass
        with engine.use(Model.LAMA):
            pass
    assert engine.unloads == 0


def test_cpu_models_are_never_unloaded_for_vram():
    device = FakeDevice()
    specs = {
        Model.TEXT_LINES: Spec("x", lambda ctx: object(), 0, 0, 0, gpu=False),
        Model.HAYAI: Spec("x", lambda ctx: FakeModel(device, 1200), 1200, 0, 5),
    }
    engine = Engine("models", specs=specs, memory=device, reserve_mb=0)
    with engine.use(Model.TEXT_LINES):
        pass
    with engine.use(Model.HAYAI):  # can't fit; unloading the CPU model would free nothing
        pass
    assert "text_lines" in engine.status()["models"]


def test_measured_working_memory_is_reserved_next_time():
    device = FakeDevice()
    engine = _engine(device, {Model.HAYAI: (200, 10), Model.LAMA: (200, 10)})
    with engine.use(Model.LAMA) as lama:
        lama.run(work_mb=700)  # far more than its estimate
    with engine.use(Model.HAYAI):
        pass
    with engine.use(Model.LAMA):  # 400 held + 700 working: HAYAI goes
        pass
    assert set(engine.status()["models"]) == {"lama"}


def test_out_of_memory_frees_every_other_idle_model():
    class OutOfMemoryError(RuntimeError):
        pass

    device = FakeDevice()
    engine = _engine(device, {Model.HAYAI: (300, 0), Model.LAMA: (300, 0)})
    with engine.use(Model.HAYAI):
        pass
    with pytest.raises(OutOfMemoryError), engine.use(Model.LAMA):
        raise OutOfMemoryError("CUDA out of memory")
    assert set(engine.status()["models"]) == {"lama"}


def test_cpu_models_skip_the_slot():
    device = FakeDevice()
    specs = {Model.TEXT_LINES: Spec("x", lambda ctx: object(), 0, 0, 9, gpu=False)}
    engine = Engine("models", specs=specs, memory=device)
    with engine.slot.hold(Priority.BATCH):
        done = threading.Event()

        def other():
            with engine.use(Model.TEXT_LINES):
                done.set()

        threading.Thread(target=other).start()
        assert done.wait(2)  # not blocked by the held slot


# --------------------------------------------------------------------------- jobs
def test_batch_jobs_run_in_order_and_interactive_jobs_do_not_wait():
    jobs = JobQueue()
    started = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def long_batch(cancel):
        started.set()
        release.wait(5)
        order.append("batch1")

    first = jobs.submit(long_batch, label="chapter 1")
    second = jobs.submit(lambda cancel: order.append("batch2"), label="chapter 2")
    started.wait(2)
    click = jobs.submit(lambda cancel: "ocr done", Priority.INTERACTIVE)
    assert click.result(2) == "ocr done"  # while chapter 1 still runs
    assert first.status == "running" and second.status == "queued"
    release.set()
    second.result(5)
    assert order == ["batch1", "batch2"]
    jobs.close()


def test_cancelling_jobs():
    jobs = JobQueue()
    started = threading.Event()

    def batch(cancel):
        started.set()
        while not cancel.is_cancelled():
            time.sleep(0.01)
        return "stopped"

    running = jobs.submit(batch)
    queued = jobs.submit(lambda cancel: "never")
    started.wait(2)
    queued.cancel()
    running.cancel()
    assert running.result(2) == "stopped" and running.status == "cancelled"
    assert queued.status == "cancelled"
    jobs.close()


# --------------------------------------------------------------------------- fetching
def _no_network(*args, **kwargs):
    raise AssertionError("network used")


def test_local_files_are_used_without_the_network(tmp_path, monkeypatch):
    try:
        import huggingface_hub
    except ImportError:  # the light environment: not even importable
        pass
    else:
        monkeypatch.setattr(huggingface_hub, "snapshot_download", _no_network)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_network)
    rev = "a" * 40
    repo = tmp_path / "ocr" / "model"
    meta = repo / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    for f in ("model.safetensors", "config.json"):
        (repo / f).write_text("x")
        (meta / f"{f}.metadata").write_text(f"{rev}\nhash\n1\n")
    single = tmp_path / "detection" / "w.pt"
    single.parent.mkdir(parents=True)
    single.write_text("x")
    fetch = Fetcher(tmp_path, token=None, hf_cache=False, offline=False)
    assert fetch.hf_repo("o/r", repo, revision=rev, files=("model.safetensors", "config.json")) == repo
    assert fetch.hf_file("o/r", "model.pt", single) == single


def test_files_from_another_revision_are_downloaded_again(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: calls.append(k["revision"]))
    repo = tmp_path / "model"
    meta = repo / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (repo / "model.safetensors").write_text("x")
    (meta / "model.safetensors.metadata").write_text("b" * 40 + "\n")
    Fetcher(tmp_path, None, False, False).hf_repo(
        "o/r", repo, revision="a" * 40, files=("model.safetensors",)
    )
    assert calls == ["a" * 40]


def test_offline_never_downloads(tmp_path):
    from mangatl.common.exceptions import ModelError

    with pytest.raises(ModelError, match="offline"):
        Fetcher(tmp_path, None, False, offline=True).hf_file("o/r", "m.pt", tmp_path / "m.pt")
