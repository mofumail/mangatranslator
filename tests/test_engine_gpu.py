"""The engine on the GPU (#11 acceptance): warm region calls are interactive-fast, and a batch job
and interactive requests share one engine without changing the batch's pages.

Needs the models, a GPU and a local harness run of om10 (see test_stages_gpu); skipped otherwise.
"""

import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from tests.test_stages_gpu import PAGE, _bubble_ids, _source_project  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def setup():
    import json

    from mangatl.common import eval_hooks
    from mangatl.engine import Engine
    from mangatl.project.store import Project
    from mangatl.settings import DebugSettings

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    src = _source_project()
    if src is None:
        pytest.skip("no stored om10 project (run eval/harness.py check --only om10)")
        raise AssertionError
    stored = Project(src, create=False)
    settings = json.loads(stored.get_settings(stored.load_page(PAGE).settings))
    stored.close()
    if not settings["debug"].get("llm_replay"):
        pytest.skip("the stored project doesn't replay a baseline cache")
    eval_hooks.configure(DebugSettings(**settings["debug"]))
    engine = Engine(settings["runtime"]["models_dir"])
    yield src, engine
    engine.close()


def _quiet(w):
    w.config = w.config.updated(**{"runtime.verbose": False})  # the run logged verbosely


def _copy(src: Path, tmp_path: Path):
    from mangatl.project.store import Project

    shutil.copy(src, tmp_path / "project.mtl")
    return Project(tmp_path / "project.mtl", create=False)


def test_warm_region_calls_take_under_300_ms(setup, tmp_path):
    from mangatl.common.caching import get_cache
    from mangatl.engine import Priority
    from mangatl.pipeline import stages, stored

    src, engine = setup
    project = _copy(src, tmp_path)
    w = stored.load(project, PAGE, engine.models(Priority.INTERACTIVE))
    _quiet(w)
    rid = _bubble_ids(project.load_page(PAGE), 1)[0]
    stages.ocr(w, [rid])  # warm: models loaded, page lines found
    stages.typeset(w)
    stages.render(w)

    def best_of(n, fn) -> float:
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    def ocr():
        get_cache().clear_all()  # a new crop, not a cached read
        stages.ocr(w, [rid])

    def retypeset():
        stages.typeset(w)
        stages.render(w)

    assert best_of(3, ocr) < 0.3
    assert best_of(3, retypeset) < 0.3
    project.close()


def test_a_batch_and_interactive_requests_share_the_engine(setup, tmp_path):
    """A chapter runs as a batch job while a person re-reads and re-typesets a page of another
    project: the requests finish while the batch runs, and the batch's pages come out exactly as
    they do alone (the harness run)."""
    import json

    import numpy as np
    from PIL import Image

    from mangatl.cli import run
    from mangatl.common.caching import get_cache
    from mangatl.engine import Priority
    from mangatl.pipeline import stages, stored
    from mangatl.settings import Settings

    src, engine = setup
    project = _copy(src, tmp_path)
    page = project.load_page(PAGE)
    out = tmp_path / "batch"
    config = Settings.model_validate_json(project.get_settings(page.settings)).resolve_secrets()
    config = config.updated(
        **{"output.project": out / "project.mtl", "translation.glossary_path": out / "glossary.txt"}
    )
    chapter = ROOT / "eval" / "in" / "tojime_no_siora"
    job = engine.jobs.submit(
        lambda cancel: run(config.with_models(engine.models(Priority.BATCH)), chapter, out, True, "", cancel),
        label="chapter",
    )

    w = stored.load(project, PAGE, engine.models(Priority.INTERACTIVE))
    _quiet(w)
    rid = _bubble_ids(page, 1)[0]
    reference = page.region(rid).ocr
    waits = []
    while job.status in ("queued", "running"):
        t0 = time.perf_counter()
        get_cache().clear_all()
        stages.ocr(w, [rid])
        stages.typeset(w)
        stages.render(w)
        waits.append(time.perf_counter() - t0)
        assert w.ocr_texts[[i["region_id"] for i in w.ordered].index(rid)] == reference
    assert job.result() == 0
    assert len(waits) >= 5, "the batch finished before the requests could interleave"
    # a request waits for at most one model call of the batch (the longest: a page's text map)
    assert sorted(waits)[len(waits) // 2] < 1.5, waits

    expected = src.parent
    written = sorted(p.name for p in out.glob("*_translated.png"))
    assert written == sorted(p.name for p in expected.glob("*_translated.png"))
    for name in written:
        a = np.asarray(Image.open(out / name))
        b = np.asarray(Image.open(expected / name))
        assert a.shape == b.shape and np.array_equal(a, b), name
    print(json.dumps(engine.status(), indent=1))
    project.close()
