"""Page documents: the operation log, undo, conflicts, staleness, proposals, blobs."""

import threading

import numpy as np
import pytest
from PIL import Image

from mangatl.project import blobs
from mangatl.project import ops as oplog
from mangatl.project.model import Page, Region, SourceRef
from mangatl.project.store import Project


def _page(n_regions: int = 2) -> Page:
    regions = [
        Region(
            id=f"r{i}",
            order=i - 1,
            kind="bubble",
            bbox=(10 * i, 0, 10 * i + 8, 8),
            confidence=0.9,
            ocr=f"src{i}",
            translation=f"en{i}",
            status={"detect": "done", "ocr": "done", "translate": "done", "typeset": "done"},
        )
        for i in range(1, n_regions + 1)
    ]
    return Page(
        id="ch1/001.jpg",
        source=SourceRef(path="ch1/001.jpg", blob="0" * 64, w=100, h=100),
        work_mode="RGB",
        canvas_mode="RGB",
        target_mode="RGB",
        regions=regions,
        next_region=n_regions + 1,
        settings="s",
    )


@pytest.fixture
def project(tmp_path):
    p = Project(tmp_path / "p.mtl")
    p.save_page(_page())
    yield p
    p.close()


def test_edit_marks_downstream_stale_and_undo_restores(project):
    doc = project.load_page("ch1/001.jpg").model_dump(mode="json")
    ops = oplog.set_op(doc, ("regions", "r1", "ocr"), "fixed")
    gid = project.apply("ch1/001.jpg", ops, label="fix OCR")
    r1 = project.load_page("ch1/001.jpg").region("r1")
    assert r1.ocr == "fixed"
    assert r1.status == {"detect": "done", "ocr": "user", "translate": "stale", "typeset": "stale"}
    project.undo_group(gid)
    r1 = project.load_page("ch1/001.jpg").region("r1")
    assert r1.ocr == "src1" and r1.status["translate"] == "done" and r1.status["ocr"] == "done"


def test_undo_one_operation_of_many(project):
    pid = "ch1/001.jpg"
    doc = project.load_page(pid).model_dump(mode="json")
    project.apply(pid, oplog.set_op(doc, ("regions", "r1", "translation"), "A"))
    doc = project.load_page(pid).model_dump(mode="json")
    project.apply(pid, oplog.set_op(doc, ("regions", "r2", "translation"), "B"))
    first = project.history(pid)[1]["id"]
    seq = project.group_ops(first)[0][0]
    project.undo_op(seq)  # only the r1 change, the later r2 change stays
    page = project.load_page(pid)
    assert page.region("r1").translation == "en1" and page.region("r2").translation == "B"


def test_undo_conflicts_with_later_change(project):
    pid = "ch1/001.jpg"
    doc = project.load_page(pid).model_dump(mode="json")
    gid = project.apply(pid, oplog.set_op(doc, ("regions", "r1", "translation"), "A"))
    doc = project.load_page(pid).model_dump(mode="json")
    project.apply(pid, oplog.set_op(doc, ("regions", "r1", "translation"), "B"))
    with pytest.raises(oplog.Conflict):
        project.undo_group(gid)
    assert project.load_page(pid).region("r1").translation == "B"  # untouched


def test_region_ids_are_stable_when_order_and_membership_change(project):
    pid = "ch1/001.jpg"
    doc = project.load_page(pid).model_dump(mode="json")
    new = Region(id="r3", order=0, kind="outside", bbox=(0, 50, 9, 60), confidence=0.5).model_dump(
        mode="json"
    )
    ops = oplog.set_op(doc, ("regions", "r3"), new)
    ops += oplog.set_op(doc, ("regions", "r1", "order"), 1)
    ops += oplog.set_op(doc, ("regions", "r2", "order"), 2)
    ops += oplog.set_op(doc, ("next_region",), 4)
    gid = project.apply(pid, ops)
    doc = project.load_page(pid).model_dump(mode="json")
    project.apply(pid, oplog.set_op(doc, ("regions", "r2", "translation"), "kept"))
    page = project.load_page(pid)
    assert [r.id for r in sorted(page.regions, key=lambda r: r.order)] == ["r3", "r1", "r2"]
    # removing a region doesn't disturb the others' ids or edits
    doc = page.model_dump(mode="json")
    project.apply(pid, oplog.set_op(doc, ("regions", "r1"), None))
    page = project.load_page(pid)
    assert {r.id for r in page.regions} == {"r2", "r3"} and page.region("r2").translation == "kept"
    with pytest.raises(oplog.Conflict):  # the order change of r1 can't be undone: r1 is gone
        project.undo_group(gid)


def test_proposals_are_accepted_or_rejected_per_operation(project):
    pid = "ch1/001.jpg"
    doc = project.load_page(pid).model_dump(mode="json")
    ops = [
        oplog.Op(("regions", "r1", "translation"), "en1", "P1"),
        oplog.Op(("regions", "r2", "translation"), "en2", "P2"),
    ]
    gid = project.propose(pid, ops)
    assert project.load_page(pid).region("r1").translation == "en1"  # not applied yet
    (s1, _), (s2, _) = project.group_ops(gid)
    project.accept(gid, [s1])
    project.reject(gid, [s2])
    page = project.load_page(pid)
    assert page.region("r1").translation == "P1" and page.region("r2").translation == "en2"
    assert project.history(pid)[-1]["status"] == "applied"
    # a proposal made against an older state is refused
    gid = project.propose(pid, [oplog.Op(("regions", "r2", "translation"), "stale value", "P3")])
    with pytest.raises(oplog.Conflict):
        project.accept(gid)
    del doc


def test_invalid_edits_are_rejected_atomically(project):
    pid = "ch1/001.jpg"
    with pytest.raises(Exception):
        project.apply(pid, [oplog.Op(("regions", "r1", "kind"), "bubble", "nonsense")])
    assert project.load_page(pid).region("r1").kind == "bubble"
    assert len(project.history(pid)) == 1


def test_masks_and_patches_round_trip(project):
    mask = np.zeros((40, 30), np.uint8)
    mask[5:9, 3:20] = 255
    ref = blobs.put_mask(project, mask)
    assert (ref.x, ref.y, ref.w, ref.h) == (3, 5, 17, 4)
    back = blobs.get_mask(project, ref)
    assert back.dtype == np.uint8 and np.array_equal(back, mask)
    as_bool = blobs.get_mask(project, blobs.put_mask(project, mask > 0))
    assert as_bool.dtype == np.bool_ and np.array_equal(as_bool, mask > 0)
    empty = blobs.get_mask(project, blobs.put_mask(project, np.zeros((7, 9), np.uint8)))
    assert empty.shape == (7, 9) and not empty.any()
    with pytest.raises(ValueError):
        blobs.put_mask(project, np.array([[0, 1, 2]], np.uint8))

    rng = np.random.default_rng(0)
    base = Image.fromarray(rng.integers(0, 255, (200, 120, 4), dtype=np.uint8), "RGBA")
    edited = base.copy()
    edited.paste((1, 2, 3, 255), (10, 10, 30, 20))
    edited.paste((9, 9, 9, 255), (100, 150, 110, 190))
    patches = blobs.diff_patches(project, base, edited)
    assert len(patches) == 2
    assert np.array_equal(np.asarray(blobs.apply_patches(project, base, patches)), np.asarray(edited))
    img = blobs.get_image(project, blobs.put_image(project, base))
    assert img.mode == "RGBA" and np.array_equal(np.asarray(img), np.asarray(base))


def test_parallel_pages_write_concurrently(tmp_path):
    project = Project(tmp_path / "p.mtl")

    def save(i):
        page = _page().model_copy(update={"id": f"{i:03}.jpg"})
        project.save_page(page)
        blobs.put_mask(project, np.eye(20, dtype=np.uint8) * 255)
        project.close()

    threads = [threading.Thread(target=save, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert project.page_ids() == [f"{i:03}.jpg" for i in range(8)]
    reopened = Project(tmp_path / "p.mtl")
    assert reopened.load_page("003.jpg").region("r2").translation == "en2"


def test_capture_checks_every_region_key_typesetting_reads():
    """capture's round-trip check compares READ_KEYS; a key typesetting starts reading must be
    added there (and to the page document), or rebuilt pages could silently differ."""
    import ast
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "mangatl"
    tree = ast.parse((root / "project/capture.py").read_text())
    read_keys = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "READ_KEYS"
    )
    pattern = re.compile(r'\b(?:bubble|item|b)(?:\.get\(|\.pop\(|\[)"(\w+)"|"(\w+)" in (?:bubble|item|b)\b')
    used = set()
    for f in ("typeset/page.py", "typeset/narration.py", "typeset/sfx_labels.py", "clean/outside_text.py"):
        text = (root / f).read_text()
        if f == "clean/outside_text.py":  # only the fills painted at render time
            text = text[text.index("def paint_pending_osb_fill(") : text.index("def _osb_fill_boxes(")]
        used |= {a or b for a, b in pattern.findall(text)}
    written_by_render = {"translation", "_sfx_label"}
    assert used - written_by_render <= set(read_keys), used - written_by_render - set(read_keys)
