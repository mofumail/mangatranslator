"""Stored pages in and out of the stages: what the editor and the agent work with.

    w = load(project, "ch1/002.jpg")          # a WorkPage, as the pipeline left it
    stages.ocr(w, ["r7"])                     # re-read one region
    stages.translate(w, ["r7"])               # same request (page context), only r7 changes
    stages.typeset(w); image = stages.render(w)
    save(project, "ch1/002.jpg", w, ran={"ocr", "translate"}, region_ids=["r7"])

`save` records the differences from the stored document as one group of operations (undoable,
statuses following what changed). The images a stage needs and the document doesn't hold are
rebuilt on load: the OCR/LLM crops (from the work image, masks and boxes, as prepare() cut them)
and the outside-text line OCR's page.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mangatl.common.image_utils import pil_to_cv2
from mangatl.detect.outside_text import outside_text_crop
from mangatl.ocr import line_ocr
from mangatl.pipeline.batch import natural_path_sort_key
from mangatl.pipeline.page import _resolve_processing_scale
from mangatl.pipeline.stages import (
    WorkPage,
    _encode_full_page_context,
    _prepare_bubble_crops,
    get_image_encoding_params,
)
from mangatl.project import blobs
from mangatl.project import ops as oplog
from mangatl.project.capture import (
    _decode_source,
    _item,
    _processed,
    canvas_image,
    page_document,
    work_image,
)
from mangatl.project.model import Page, Region
from mangatl.project.store import Author, Project
from mangatl.settings import Settings
from mangatl.translate.page import pair_context_lines


def _detect_image(project: Project, page: Page, work: Image.Image) -> Image.Image:
    """The page detection saw (before any inpainting): the work image, unless outside text was
    inpainted into it at detection (no overlap), then the source as decoded for it."""
    if page.work_image is not None or not page.work_patches:
        return work
    src = page.source
    crop = (src.crop_mode, src.crop) if src.crop is not None and src.crop_mode is not None else None
    return _decode_source(project.get_blob(src.blob), page.work_mode, crop)


def _region_dict(project: Project, region: Region, page_cv: np.ndarray) -> dict:
    item = _item(project, region, page_cv)
    item["region_id"] = region.id
    item["confidence"] = region.confidence
    if region.label is not None:
        item["class"] = region.label
    if region.panel is not None:
        item["panel_id"] = region.panel
    if region.seg_mask is not None:
        item["seg_mask"] = blobs.get_mask(project, region.seg_mask)
    if region.neighbors:
        item["conjoined_neighbor_bboxes"] = list(region.neighbors)
    if region.flag is not None:
        item["region_flag"] = region.flag
    return item


def load(project: Project, page_id: str) -> WorkPage:
    """A stored page as a WorkPage, ready for any stage after detection."""
    page = project.load_page(page_id)
    config = Settings.model_validate_json(project.get_settings(page.settings)).resolve_secrets()
    work = work_image(project, page)
    page_cv = pil_to_cv2(work)
    source_format = Image.open(io.BytesIO(project.get_blob(page.source.blob))).format
    mime_type, cv2_ext = get_image_encoding_params(source_format)
    detected = _detect_image(project, page, work)
    w = WorkPage(
        config=config,
        image=work,
        processing_scale=page.render.processing_scale
        if page.render
        else _resolve_processing_scale(work, config),
        target_mode=page.target_mode,
        mime_type=mime_type,
        cv2_ext=cv2_ext,
        image_format=source_format,
        source_path=Path(page.source.path),
        context_image=detected.copy(),
        page_cv=page_cv,
        next_region=page.next_region,
        panels=[tuple(b) for b in page.panels] if page.panels is not None else None,
        locked={r.id for r in page.regions if r.locked},
        edited={r.id for r in page.regions if "user" in r.status.values()},
    )
    if config.translation.full_page_context:  # the page image the translation request carries
        w.full_image_b64 = _encode_full_page_context(w.context_image, config, cv2_ext)
    by_id = sorted(page.regions, key=lambda r: int(r.id[1:]) if r.id[1:].isdigit() else 0)  # detection order
    items = {r.id: _region_dict(project, r, page_cv) for r in page.regions}

    # outside text: crops as detection cut them, from the page before inpainting
    outside = sorted((r for r in page.regions if r.outside is not None), key=lambda r: r.outside.detect_index)  # type: ignore[union-attr]
    if outside:
        osb_mime = "image/png" if source_format and source_format.upper() == "PNG" else "image/jpeg"
        osb_ext = ".png" if source_format and source_format.upper() == "PNG" else ".jpg"
        original_cv = cv2.cvtColor(np.array(detected), cv2.COLOR_RGB2BGR)
        token = line_ocr.register_page(detected)
        for r in outside:
            assert r.outside is not None
            crop = outside_text_crop(original_cv, r.bbox, r.outside.original_bbox, config, config.verbose)
            ok, buffer = cv2.imencode(osb_ext, crop)
            item = items[r.id]
            item.update(confidence=r.confidence, mime_type=osb_mime, mt_page=token)
            if ok:
                item["image_b64"] = base64.b64encode(buffer).decode("utf-8")
    w.outside = [items[r.id] for r in outside]

    # bubbles: crops cut together (each whites out its conjoined neighbours)
    bubbles = [items[r.id] for r in by_id if r.outside is None]
    w.bubbles = _prepare_bubble_crops(bubbles, [], page_cv, config, mime_type) if bubbles else []
    prepared = {item["region_id"]: item for item in w.bubbles}
    items.update(prepared)

    ordered = sorted((r for r in page.regions if r.order is not None), key=lambda r: r.order or 0)
    w.ordered = [items[r.id] for r in ordered]
    w.translations = [r.translation or "" for r in ordered]
    w.ocr_texts = [r.ocr or "" for r in ordered]
    w.cleaning = [_processed(project, r.cleaning) for r in by_id if r.cleaning is not None]
    w.canvas = canvas_image(project, page, work)
    if page.render is not None:
        from mangatl.typeset.page import _RenderSizes

        rs = page.render
        w.sizes = _RenderSizes(**{name: getattr(rs, name) for name in _RenderSizes._fields})
        if rs.glyphs is not None:
            w.glyphs = {"page": rs.glyphs.page, "own": {g.bbox: (g.size, g.parts) for g in rs.glyphs.bubbles}}
        w.typeset_done = True
    return w


def _page_lines(project: Project, page: Page) -> list[str]:
    """A page's text as later pages see it: each region in reading order as "source → translation"
    (a strip: its sections' lines in order)."""
    if page.sections:
        return [line for sid in page.sections for line in _page_lines(project, project.load_page(sid))]
    ordered = sorted((r for r in page.regions if r.order is not None), key=lambda r: r.order or 0)
    if not ordered or page.render is None:
        return []
    return pair_context_lines([r.ocr or "" for r in ordered], [r.translation or "" for r in ordered])


def previous_texts(project: Project, page_id: str, count: int) -> list[list[str]]:
    """The previous `count` pages' text (in reading order of the pages, as the batch run sends
    it): the story context of a page's translation request."""
    if count <= 0:
        return []
    ids = sorted(
        (pid for pid in project.page_ids() if "#" not in pid), key=lambda p: natural_path_sort_key(Path(p))
    )
    if page_id not in ids:
        return []
    before = ids[max(0, ids.index(page_id) - count) : ids.index(page_id)]
    pages = [_page_lines(project, project.load_page(pid)) for pid in before]
    return [lines for lines in pages if lines]


def save(
    project: Project,
    page_id: str,
    w: WorkPage,
    author: Author = "pipeline",
    label: str = "edit",
    ran: Iterable[str] = (),
    region_ids: Iterable[str] | None = None,
) -> int | None:
    """Record the working page's differences from the stored one as one group of operations
    (None when nothing changed). `ran`: the stages (re)computed for `region_ids` (None: all
    regions); their status becomes done (or user), later stages that weren't rerun go stale."""
    stored = project.load_page(page_id)
    src = stored.source
    crop = (src.crop_mode, src.crop) if src.crop is not None and src.crop_mode is not None else None
    new = page_document(
        project, page_id, work=w, source_path=Path(src.path), source_crop=crop, output_path=stored.output
    )
    # the stages ran on the regions asked for, never on a locked one (its status stays a person's)
    scope = ({r.id for r in new.regions} if region_ids is None else set(region_ids)) - w.locked
    stages = set(ran)
    ops = oplog.diff(
        stored.model_dump(mode="json"),
        new.model_dump(mode="json"),
        author,
        {rid: stages for rid in scope},
    )
    if not ops:
        return None
    return project.apply(page_id, ops, author=author, label=label)
