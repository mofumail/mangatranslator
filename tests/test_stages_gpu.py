"""The stage API on a stored page (#10 acceptance): region-scoped calls change only their regions.

Needs the models, a GPU-capable environment and a local harness run of om10 (a project.mtl
whose settings replay the baseline's LLM answers); skipped otherwise (CI).
"""

import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytest.importorskip("cv2")
pytest.importorskip("torch")

PAGE = "002.jpg"  # the first page: its request carries no previous pages and an empty glossary


def _source_project() -> Path | None:
    """The newest default-settings harness run of om10 whose project stores panels (projects
    from before the stage API don't, and their reading order can't be reproduced)."""
    from mangatl.project.store import Project

    runs = sorted(
        ROOT.glob("eval/runs/*def/om10/tojime_no_siora/out/project.mtl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        project = Project(run, create=False)
        try:
            if project.load_page(PAGE).panels is not None:
                return run
        finally:
            project.close()
    return None


@pytest.fixture(scope="module")
def env():
    import json

    from mangatl.common import eval_hooks
    from mangatl.ml.model_manager import get_model_manager
    from mangatl.project.store import Project
    from mangatl.settings import DebugSettings

    src = _source_project()
    if src is None:
        pytest.skip("no stored om10 project (run eval/harness.py check --only om10)")
        raise AssertionError  # unreachable (for the type checker)
    stored = Project(src, create=False)
    settings = json.loads(stored.get_settings(stored.load_page(PAGE).settings))
    stored.close()
    debug = settings["debug"]
    if not debug.get("llm_replay") or not Path(debug.get("llm_cache") or "").is_dir():
        pytest.skip("the stored project doesn't replay a baseline cache")
    eval_hooks.configure(DebugSettings(**debug))
    get_model_manager().set_models_dir(settings["runtime"]["models_dir"])
    return src


@pytest.fixture
def project(env, tmp_path):
    from mangatl.project.store import Project

    shutil.copy(env, tmp_path / "project.mtl")
    p = Project(tmp_path / "project.mtl", create=False)
    yield p
    p.close()


def _bubble_ids(page, n=2):
    return [
        r.id
        for r in sorted(page.regions, key=lambda r: r.order if r.order is not None else 999)
        if r.kind == "bubble" and r.order is not None
    ][:n]


def _json(region) -> str:
    return region.model_dump_json()


def test_reocr_one_region_leaves_the_others_untouched(project):
    from mangatl.pipeline import stages, stored
    from mangatl.project import ops as oplog

    page = project.load_page(PAGE)
    rid = _bubble_ids(page, 1)[0]
    doc = page.model_dump(mode="json")
    project.apply(PAGE, oplog.set_op(doc, ("regions", rid, "ocr"), "wrong read", author="user"))
    before = {r.id: _json(r) for r in project.load_page(PAGE).regions}

    w = stored.load(project, PAGE)
    stages.ocr(w, [rid])
    assert stored.save(project, PAGE, w, ran={"ocr"}, region_ids=[rid]) is not None

    after = project.load_page(PAGE)
    region = after.region(rid)
    assert region.ocr == page.region(rid).ocr  # read again from the crop
    assert region.status["ocr"] == "done" and region.status["translate"] == "stale"
    for r in after.regions:
        if r.id != rid:
            assert _json(r) == before[r.id], r.id


def test_edited_translation_changes_only_that_regions_pixels(project):
    import numpy as np
    from PIL import Image

    from mangatl.pipeline import stages, stored
    from mangatl.project import ops as oplog

    page = project.load_page(PAGE)
    rid = _bubble_ids(page, 1)[0]
    written = Image.open(page.output)
    written.load()
    project.apply(
        PAGE, oplog.set_op(page.model_dump(mode="json"), ("regions", rid, "translation"), "Hello there!")
    )

    w = stored.load(project, PAGE)
    stages.typeset(w)
    image = stages.render(w).convert(w.target_mode)
    a, b = np.asarray(image), np.asarray(written.convert(image.mode))
    changed = (a != b).any(axis=-1) if a.ndim == 3 else a != b
    assert changed.any()
    ys, xs = np.nonzero(changed)
    x1, y1, x2, y2 = page.region(rid).bbox
    margin = 8  # text outline / antialiasing
    assert (
        xs.min() >= x1 - margin
        and xs.max() < x2 + margin
        and ys.min() >= y1 - margin
        and ys.max() < y2 + margin
    )

    stored.save(project, PAGE, w, ran={"typeset"}, region_ids=[rid])
    assert project.load_page(PAGE).region(rid).status["typeset"] == "done"


def test_region_scoped_translate_sends_the_page_and_writes_one_region(project, tmp_path):
    from mangatl.pipeline import stages, stored
    from mangatl.project import ops as oplog

    page = project.load_page(PAGE)
    keep, redo = _bubble_ids(page, 2)
    doc = page.model_dump(mode="json")
    ops = oplog.set_op(doc, ("regions", keep, "translation"), "KEEP ME", author="user")
    oplog.apply(doc, ops)
    ops2 = oplog.set_op(doc, ("regions", redo, "translation"), "REDO ME", author="user")
    project.apply(PAGE, ops + ops2)

    w = stored.load(project, PAGE)
    empty = tmp_path / "glossary.txt"
    empty.write_text("")  # as it was when the page was first translated
    w.config = w.config.updated(**{"translation.glossary_path": empty})
    stages.translate(
        w, stored.previous_texts(project, PAGE, w.config.translation.context_pages), region_ids=[redo]
    )
    stored.save(project, PAGE, w, ran={"translate"}, region_ids=[redo])

    after = project.load_page(PAGE)
    assert after.region(redo).translation == page.region(redo).translation  # the same request, replayed
    assert after.region(keep).translation == "KEEP ME"


def test_locked_regions_are_left_alone(project):
    from mangatl.pipeline import stages, stored
    from mangatl.project import ops as oplog

    page = project.load_page(PAGE)
    rid = _bubble_ids(page, 1)[0]
    doc = page.model_dump(mode="json")
    ops = oplog.set_op(doc, ("regions", rid, "ocr"), "my reading", author="user")
    oplog.apply(doc, ops)
    ops += oplog.set_op(doc, ("regions", rid, "locked"), True, author="user")
    project.apply(PAGE, ops)

    w = stored.load(project, PAGE)
    stages.ocr(w)  # every region
    stored.save(project, PAGE, w, ran={"ocr"})
    region = project.load_page(PAGE).region(rid)
    assert region.ocr == "my reading" and region.locked and region.status["ocr"] == "user"


def _rerun_after_redetect(w, found, project, glossary):
    from mangatl.pipeline import stages, stored

    redo = found.moved + found.added
    stages.prepare(w)
    glossary.write_text("")  # as it was when the page was first translated
    w.config = w.config.updated(**{"translation.glossary_path": glossary})
    if redo:
        stages.ocr(w, redo)
        stages.translate(
            w, stored.previous_texts(project, PAGE, w.config.translation.context_pages), region_ids=redo
        )
    stages.typeset(w)


def test_redetecting_an_unchanged_page_keeps_every_region(project, tmp_path):
    import numpy as np
    from PIL import Image

    from mangatl.pipeline import stages, stored

    page = project.load_page(PAGE)
    w = stored.load(project, PAGE)
    found = stages.redetect(w)
    assert sorted(found.same) == sorted(r.id for r in page.regions)
    assert not (found.moved or found.added or found.removed or found.kept)
    stages.clean(w)
    _rerun_after_redetect(w, found, project, tmp_path / "glossary.txt")
    written = Image.open(page.output)
    written.load()
    image = stages.render(w).convert(w.target_mode)
    assert np.array_equal(np.asarray(image), np.asarray(written.convert(image.mode)))
    stored.save(project, PAGE, w, ran={"detect", "clean", "typeset"})
    after = project.load_page(PAGE)
    assert [r.id for r in after.regions] == [r.id for r in page.regions]


def test_a_region_found_again_gets_a_new_id_and_the_others_keep_theirs(project, tmp_path):
    import numpy as np
    from PIL import Image

    from mangatl.pipeline import stages, stored
    from mangatl.project import ops as oplog

    page = project.load_page(PAGE)
    gone = _bubble_ids(page, 1)[0]
    box = page.region(gone).bbox
    project.apply(PAGE, oplog.set_op(page.model_dump(mode="json"), ("regions", gone), None))
    before = {r.id: r.model_dump_json() for r in project.load_page(PAGE).regions}

    w = stored.load(project, PAGE)
    margin = 10
    found = stages.redetect(w, area=(box[0] - margin, box[1] - margin, box[2] + margin, box[3] + margin))
    assert found.added == [f"r{page.next_region}"] and not found.removed
    stages.clean(w, found.added)
    _rerun_after_redetect(w, found, project, tmp_path / "glossary.txt")
    written = Image.open(page.output)
    written.load()
    image = stages.render(w).convert(w.target_mode)
    assert np.array_equal(np.asarray(image), np.asarray(written.convert(image.mode)))  # the page as it was
    stored.save(
        project, PAGE, w, ran={"detect", "clean", "ocr", "translate", "typeset"}, region_ids=found.added
    )

    after = project.load_page(PAGE)
    new = after.region(found.added[0])
    assert new.bbox == box and new.translation == page.region(gone).translation
    assert {r.id for r in after.regions} == set(before) | set(found.added)
    for r in after.regions:
        if r.id in before:
            assert r.model_dump_json() == before[r.id], r.id


def test_a_vanished_region_with_a_persons_work_is_kept_and_flagged(project):
    from mangatl.pipeline import stages, stored
    from mangatl.project import ops as oplog
    from mangatl.project.model import Region

    page = project.load_page(PAGE)
    ghost = Region(
        id=f"r{page.next_region}",
        order=None,
        kind="bubble",
        bbox=(1, 1, 9, 9),
        confidence=1.0,
        translation="mine",
    ).model_dump(mode="json")
    doc = page.model_dump(mode="json")
    ops = oplog.set_op(doc, ("regions", ghost["id"]), ghost)
    oplog.apply(doc, ops)
    ops += oplog.set_op(doc, ("regions", ghost["id"], "translation"), "my words", author="user")
    oplog.apply(doc, ops[-1:])
    ops += oplog.set_op(doc, ("next_region",), page.next_region + 1)
    project.apply(PAGE, ops)

    w = stored.load(project, PAGE)
    found = stages.redetect(w)
    assert found.kept == [ghost["id"]] and not found.removed
