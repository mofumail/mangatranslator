"""The pipeline's page state in and out of a page document.

The stages still pass dicts between them (the stage API, #10, will make them read and write the
page document directly). Until then, `capture` turns what typesetting gets (the work image, the
cleaned canvas, the regions in reading order, their translations and cleaning results, the page's
sizes) into a Page, and `render_inputs` turns a Page back into exactly those arguments, so that
`rebuild` renders the same page from the project file alone.

Capture checks itself: every part it stores without a blob (the work image as the source plus
patches, the cleaned canvas as patches, original lettering cut from the work image) is rebuilt on
the spot and compared with the live value; a part that doesn't match is stored whole instead.
"""

from __future__ import annotations

import dataclasses
import io
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mangatl.common.image_utils import convert_image_to_target_mode, cv2_to_pil, pil_to_cv2
from mangatl.project import blobs
from mangatl.project.model import (
    BubbleCleaning,
    Glyphs,
    GlyphSize,
    OriginalCrop,
    OutsideText,
    Page,
    PendingFill,
    Region,
    RenderState,
    SourceRef,
)
from mangatl.project.store import Project
from mangatl.settings import Settings

_projects: dict[Path, Project] = {}
_projects_lock = threading.Lock()


def open_project(path: str | Path) -> Project:
    """The process's Project for `path` (one per file; its connections are per thread)."""
    key = Path(path).resolve()
    with _projects_lock:
        if key not in _projects:
            _projects[key] = Project(key)
        return _projects[key]


def close_projects() -> None:
    """Fold the write-ahead log into the project files, so each is one self-contained file."""
    with _projects_lock:
        for project in _projects.values():
            project.checkpoint()
            project.close()
        _projects.clear()


# --- small exact conversions ----------------------------------------------------------------------
def _int(v: Any) -> int:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
        raise TypeError(f"expected an int, got {type(v).__name__} {v!r}")
    return int(v)


def _ints(values: Any) -> tuple[int, ...]:
    return tuple(_int(v) for v in values)


def _num(v: Any) -> int | float:
    v = v.item() if isinstance(v, np.generic) else v
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise TypeError(f"expected a number, got {type(v).__name__} {v!r}")
    return v


def _same(a: Image.Image, b: Image.Image) -> bool:
    return a.mode == b.mode and a.size == b.size and np.array_equal(np.asarray(a), np.asarray(b))


def _crop_box(bbox: Any, w: int, h: int) -> tuple[int, int, int, int]:
    """The clamped crop the pipeline cuts for a region's original lettering."""
    x1, y1, x2, y2 = (round(c) for c in bbox)
    return max(0, min(w, x1)), max(0, min(h, y1)), max(0, min(w, x2)), max(0, min(h, y2))


def _cut(page_cv: np.ndarray, bbox: Any, mode: str) -> Image.Image:
    h, w = page_cv.shape[:2]
    x1, y1, x2, y2 = _crop_box(bbox, w, h)
    crop = cv2_to_pil(page_cv[y1:y2, x1:x2].copy())
    return crop if crop.mode == mode else crop.convert(mode)


def _decode_source(data: bytes, mode: str, crop: SourceCrop | None = None) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    if crop is not None:  # a strip's section: cut from the strip as the strip was decoded
        crop_mode, box = crop
        image = convert_image_to_target_mode(image, crop_mode).crop(box)
    return convert_image_to_target_mode(image, mode)


SourceCrop = tuple[str, tuple[int, int, int, int]]  # (mode the file is decoded in, box)


# --- capture ------------------------------------------------------------------------------------
@dataclass
class _Images:
    """What capture needs to store a page's images."""

    source_path: Path
    source_crop: SourceCrop | None
    work: Image.Image  # the page as the pipeline worked on it (page_cv is this, in OpenCV order)
    canvas: Image.Image  # what text is drawn on (or, unrendered, the result)


def _store_images(project: Project, page_id: str, images: _Images) -> dict[str, Any]:
    data = images.source_path.read_bytes()
    source_blob = project.put_blob(data, "application/octet-stream")
    source = _decode_source(data, images.work.mode, images.source_crop)
    crop = images.source_crop
    fields: dict[str, Any] = {
        "source": SourceRef(
            path=str(images.source_path),
            blob=source_blob,
            w=source.width,
            h=source.height,
            crop=crop[1] if crop else None,
            crop_mode=crop[0] if crop else None,
        ),
        "work_mode": images.work.mode,
    }
    if source.size == images.work.size:
        fields["work_patches"] = blobs.diff_patches(project, source, images.work)
        if not _same(blobs.apply_patches(project, source, fields["work_patches"]), images.work):
            raise RuntimeError(f"{page_id}: work image patches don't rebuild the work image")
    else:  # pre-upscaled
        fields["work_image"] = blobs.put_image(project, images.work)
    base = images.work if images.work.mode == images.canvas.mode else images.work.convert(images.canvas.mode)
    fields["canvas_mode"] = images.canvas.mode
    fields["cleaned"] = blobs.diff_patches(project, base, images.canvas)
    if not _same(blobs.apply_patches(project, base, fields["cleaned"]), images.canvas):
        raise RuntimeError(f"{page_id}: cleaning patches don't rebuild the canvas")
    return fields


def _settings_id(project: Project, config: Settings) -> str:
    return project.put_settings(config.to_json())  # secrets are excluded from the JSON


def _cleaning(project: Project, info: dict) -> BubbleCleaning:
    def mask(key: str):
        m = info.get(key)
        return blobs.put_mask(project, m) if m is not None else None

    return BubbleCleaning(
        bbox=_ints(info["bbox"]),  # type: ignore[arg-type]
        mask=mask("mask"),
        base_mask=mask("base_mask"),
        art_mask=mask("art_mask"),
        color=_ints(info["color"]),  # type: ignore[arg-type]
        is_colored=bool(info.get("is_colored", False)),
        text_bbox=_ints(info["text_bbox"]) if info.get("text_bbox") is not None else None,  # type: ignore[arg-type]
        text_color_bgr=_ints(info["text_color_bgr"]) if info.get("text_color_bgr") is not None else None,  # type: ignore[arg-type]
        is_seg=bool(info.get("is_seg", False)),
        inpainted=bool(info.get("inpainted", False)),
    )


def _region(
    project: Project,
    item: dict,
    region_id: str,
    order: int | None,
    translation: str | None,
    detect_index: int | None,
    cleaning: dict | None,
    page_cv: np.ndarray,
) -> Region:
    outside = None
    if item.get("is_outside_text"):
        assert detect_index is not None
        outside = OutsideText(
            original_bbox=_ints(item["original_bbox"]),  # type: ignore[arg-type]
            dark_text=bool(item["is_dark_text"]),
            text_color=_ints(item["text_color_rgb"]) if item.get("text_color_rgb") is not None else None,  # type: ignore[arg-type]
            aspect_ratio=_num(item["aspect_ratio"]),
            needs_background=bool(item["needs_text_background"]),
            tilt_deg=_num(item["tilt_deg"]),
            tilt_conf=_num(item["tilt_conf"]),
            tilt_axis=str(item["tilt_axis"]),
            oriented_size=tuple(_num(v) for v in item["oriented_dimensions"]),  # type: ignore[arg-type]
            detect_index=detect_index,
            narration=item.get("mt_narration"),
            narration_text=item.get("mt_text"),
        )
    fill = None
    if item.get("pending_fill") is not None:
        f = item["pending_fill"]
        fill = PendingFill(
            color=_ints(f.color),  # type: ignore[arg-type]
            box=_ints(f.box),  # type: ignore[arg-type]
            tilt_deg=_num(f.tilt_deg),
            bubble_mask=blobs.put_mask(project, f.bubble_mask) if f.bubble_mask is not None else None,
        )
    original = None
    if "original_crop_pil" in item:
        crop = item["original_crop_pil"]
        original = OriginalCrop(mode=crop.mode)
        if not _same(_cut(page_cv, item["bbox"], crop.mode), crop):
            original.image = blobs.put_image(project, crop)
    seg = item.get("seg_mask")
    neighbors = item.get("conjoined_neighbor_bboxes")
    stages = ["detect", "ocr", "translate"] + (["clean"] if cleaning is not None or fill is not None else [])
    return Region(
        id=region_id,
        order=order,
        kind="outside" if outside else "bubble",
        bbox=_ints(item["bbox"]),  # type: ignore[arg-type]
        confidence=_num(item.get("confidence", 0.0)),
        label=item.get("class"),
        panel=_int(item["panel_id"]) if item.get("panel_id") is not None else None,
        seg_mask=blobs.put_mask(project, seg) if seg is not None else None,
        neighbors=[_ints(b) for b in neighbors] if neighbors else None,  # type: ignore[misc]
        ocr=item.get("ocr_text"),
        translation=translation,
        skip=bool(item.get("mt_skip", False)),
        outside=outside,
        cleaning=_cleaning(project, cleaning) if cleaning is not None else None,
        pending_fill=fill,
        original=original,
        status={s: "done" for s in stages + (["typeset"] if order is not None else [])},  # type: ignore[misc]
    )


def capture(
    project: Project,
    page_id: str,
    *,
    source_path: Path,
    target_mode: str,
    work: Image.Image,
    page_cv: np.ndarray,
    source_crop: SourceCrop | None = None,
    canvas: Image.Image,
    sorted_bubble_data: list[dict],
    translated_texts: list[str],
    processed_bubbles_info: list[dict],
    outside_text_data: list[dict],
    sizes: Any,
    processing_scale: float,
    page_glyphs: dict,
    config: Settings,
    output_path: str | Path | None,
) -> Page:
    """Record a page as typesetting is about to draw it. Call before render_translations,
    which changes the region dicts (it pops pending fills and adds labels)."""
    if not np.array_equal(pil_to_cv2(work), page_cv):
        raise RuntimeError(f"{page_id}: page_cv is not the work image")
    if len(translated_texts) != len(sorted_bubble_data):
        raise RuntimeError(
            f"{page_id}: {len(translated_texts)} translations for {len(sorted_bubble_data)} regions"
        )
    if set(page_glyphs) - {"page", "own"}:
        raise RuntimeError(f"{page_id}: unknown glyph keys {sorted(page_glyphs)}")
    fields = _store_images(project, page_id, _Images(source_path, source_crop, work, canvas))

    detect_index = {id(item): i for i, item in enumerate(outside_text_data)}
    info_by_bbox = {
        tuple(info["bbox"]): info
        for info in processed_bubbles_info
        if "bbox" in info and "color" in info and "mask" in info
    }  # what typesetting looks bubbles up in (the last of equal boxes wins)
    regions: list[Region] = []
    in_order = set()
    for order, (item, text) in enumerate(zip(sorted_bubble_data, translated_texts, strict=True)):
        outside = bool(item.get("is_outside_text"))
        if outside and id(item) not in detect_index:
            raise RuntimeError(f"{page_id}: outside text in reading order but not among the detections")
        in_order.add(id(item))
        cleaning = None if outside else info_by_bbox.get(tuple(item["bbox"]))
        regions.append(
            _region(
                project,
                item,
                f"r{len(regions) + 1}",
                order,
                text,
                detect_index.get(id(item)),
                cleaning,
                page_cv,
            )
        )
    for item in outside_text_data:  # outside text left out of the reading order (fills still paint)
        if id(item) not in in_order:
            regions.append(
                _region(
                    project, item, f"r{len(regions) + 1}", None, None, detect_index[id(item)], None, page_cv
                )
            )

    glyphs = None
    if page_glyphs:
        glyphs = Glyphs(
            page=_num(page_glyphs["page"]) if page_glyphs.get("page") is not None else None,
            bubbles=[
                GlyphSize(bbox=_ints(b), size=_num(est[0]), parts=_int(est[1]))  # type: ignore[arg-type]
                for b, est in (page_glyphs.get("own") or {}).items()
            ],
        )
    render = RenderState(
        processing_scale=_num(processing_scale),
        **{name: _num(getattr(sizes, name)) for name in sizes._fields},
        glyphs=glyphs,
    )
    page = Page(
        id=page_id,
        **fields,
        target_mode=target_mode,
        regions=regions,
        next_region=len(regions) + 1,
        render=render,
        settings=_settings_id(project, config),
        output=str(output_path) if output_path else None,
    )
    _check_round_trip(
        project,
        page,
        sorted_bubble_data,
        translated_texts,
        processed_bubbles_info,
        sizes,
        processing_scale,
        page_glyphs,
    )
    project.save_page(page)
    return page


def capture_unrendered(
    project: Project,
    page_id: str,
    *,
    source_path: Path,
    target_mode: str,
    work: Image.Image,
    result: Image.Image,
    source_crop: SourceCrop | None = None,
    config: Settings,
    output_path: str | Path | None,
) -> Page:
    """Record a page with nothing typeset (no text found, cleaning only): its result is its canvas."""
    page = Page(
        id=page_id,
        **_store_images(project, page_id, _Images(source_path, source_crop, work, result)),
        target_mode=target_mode,
        settings=_settings_id(project, config),
        output=str(output_path) if output_path else None,
    )
    project.save_page(page)
    return page


def capture_strip(
    project: Project,
    page_id: str,
    *,
    source_path: Path,
    target_mode: str,
    strip: Image.Image,
    sections: list[str],
    config: Settings,
    output_path: str | Path | None,
) -> Page:
    """Record a long strip: its source, and its sections (each a page of its own) in order."""
    data = source_path.read_bytes()
    page = Page(
        id=page_id,
        source=SourceRef(
            path=str(source_path),
            blob=project.put_blob(data, "application/octet-stream"),
            w=strip.width,
            h=strip.height,
        ),
        work_mode=strip.mode,
        canvas_mode=target_mode,
        target_mode=target_mode,
        sections=sections,
        settings=_settings_id(project, config),
        output=str(output_path) if output_path else None,
    )
    project.save_page(page)
    return page


# --- back to render inputs ------------------------------------------------------------------------
@dataclass
class RenderInputs:
    """render_translations' arguments, plus what the page flow does around it."""

    canvas: Image.Image
    sorted_bubble_data: list[dict]
    translated_texts: list[str]
    processed_bubbles_info: list[dict]
    page_cv: np.ndarray
    sizes: Any
    processing_scale: float
    page_glyphs: dict
    outside_text_data: list[dict]  # the outside text regions in detection order (same dicts)
    config: Settings
    target_mode: str


def work_image(project: Project, page: Page) -> Image.Image:
    if page.work_image is not None:
        return blobs.get_image(project, page.work_image)
    src = page.source
    crop = (src.crop_mode, src.crop) if src.crop is not None and src.crop_mode is not None else None
    source = _decode_source(project.get_blob(src.blob), page.work_mode, crop)
    return blobs.apply_patches(project, source, page.work_patches)


def canvas_image(project: Project, page: Page, work: Image.Image | None = None) -> Image.Image:
    work = work if work is not None else work_image(project, page)
    base = work if work.mode == page.canvas_mode else work.convert(page.canvas_mode)
    return blobs.apply_patches(project, base, page.cleaned)


def _item(project: Project, region: Region, page_cv: np.ndarray) -> dict:
    from mangatl.clean.outside_text import _PendingOsbFill

    item: dict[str, Any] = {"bbox": region.bbox}
    if region.ocr is not None:
        item["ocr_text"] = region.ocr
    if region.skip:
        item["mt_skip"] = True
    if region.original is not None:
        o = region.original
        item["original_crop_pil"] = (
            blobs.get_image(project, o.image) if o.image is not None else _cut(page_cv, region.bbox, o.mode)
        )
    if region.outside is not None:
        o = region.outside
        item.update(
            is_outside_text=True,
            original_bbox=o.original_bbox,
            is_dark_text=o.dark_text,
            text_color_rgb=o.text_color,
            aspect_ratio=o.aspect_ratio,
            needs_text_background=o.needs_background,
            tilt_deg=o.tilt_deg,
            tilt_conf=o.tilt_conf,
            tilt_axis=o.tilt_axis,
            oriented_dimensions=o.oriented_size,
        )
        if o.narration is not None:
            item["mt_narration"] = o.narration
        if o.narration_text is not None:
            item["mt_text"] = o.narration_text
    if region.pending_fill is not None:
        f = region.pending_fill
        mask = blobs.get_mask(project, f.bubble_mask) if f.bubble_mask is not None else None
        item["pending_fill"] = _PendingOsbFill(
            color=f.color, box=f.box, tilt_deg=f.tilt_deg, bubble_mask=mask
        )
    return item


def _processed(project: Project, c: BubbleCleaning) -> dict:
    def mask(ref):
        return blobs.get_mask(project, ref) if ref is not None else None

    return {
        "bbox": c.bbox,
        "color": c.color,
        "mask": mask(c.mask),
        "base_mask": mask(c.base_mask),
        "is_seg": c.is_seg,
        "is_colored": c.is_colored,
        "text_bbox": c.text_bbox,
        "text_color_bgr": c.text_color_bgr,
    }


def render_inputs(project: Project, page: Page) -> RenderInputs:
    """A typeset page's render arguments, rebuilt from the project file alone."""
    from mangatl.typeset.page import _RenderSizes

    if page.render is None:
        raise ValueError(f"{page.id}: nothing typeset")
    work = work_image(project, page)
    page_cv = pil_to_cv2(work)
    ordered = sorted((r for r in page.regions if r.order is not None), key=lambda r: r.order or 0)
    items = {r.id: _item(project, r, page_cv) for r in page.regions}
    processed = [_processed(project, r.cleaning) for r in ordered if r.cleaning is not None]
    outside = sorted((r for r in page.regions if r.outside is not None), key=lambda r: r.outside.detect_index)  # type: ignore[union-attr]
    rs = page.render
    glyphs: dict = {}
    if rs.glyphs is not None:
        glyphs = {"page": rs.glyphs.page, "own": {g.bbox: (g.size, g.parts) for g in rs.glyphs.bubbles}}
    return RenderInputs(
        canvas=canvas_image(project, page, work),
        sorted_bubble_data=[items[r.id] for r in ordered],
        translated_texts=[r.translation or "" for r in ordered],
        processed_bubbles_info=processed,
        page_cv=page_cv,
        sizes=_RenderSizes(**{name: getattr(rs, name) for name in _RenderSizes._fields}),
        processing_scale=rs.processing_scale,
        page_glyphs=glyphs,
        outside_text_data=[items[r.id] for r in outside],
        config=Settings.model_validate_json(project.get_settings(page.settings)),
        target_mode=page.target_mode,
    )


def _check_round_trip(
    project: Project,
    page: Page,
    sorted_bubble_data: list[dict],
    translated_texts: list[str],
    processed_bubbles_info: list[dict],
    sizes: Any,
    processing_scale: float,
    page_glyphs: dict,
) -> None:
    """The rebuilt arguments must equal the live ones in what typesetting reads (and in type:
    boxes are looked up as tuple keys)."""
    back = render_inputs(project, page)
    if back.translated_texts != list(translated_texts):
        raise RuntimeError(f"{page.id}: translations don't round-trip")
    if tuple(back.sizes) != tuple(sizes) or [type(v) for v in back.sizes] != [type(v) for v in sizes]:
        raise RuntimeError(f"{page.id}: sizes don't round-trip: {tuple(sizes)} -> {tuple(back.sizes)}")
    if back.processing_scale != processing_scale:
        raise RuntimeError(f"{page.id}: processing scale doesn't round-trip")
    if back.page_glyphs != page_glyphs:
        raise RuntimeError(f"{page.id}: glyph sizes don't round-trip")
    for live, rebuilt in zip(sorted_bubble_data, back.sorted_bubble_data, strict=True):
        for key in READ_KEYS:
            if (key in live) != (key in rebuilt) and (live.get(key), rebuilt.get(key)) != (False, None):
                raise RuntimeError(f"{page.id}: region field {key} present in only one of live/rebuilt")
            if key in rebuilt and not _equal(live[key], rebuilt[key]):
                raise RuntimeError(
                    f"{page.id}: region field {key} doesn't round-trip: {live[key]!r} -> {rebuilt[key]!r}"
                )
    live_info = _render_info(processed_bubbles_info)
    back_info = _render_info(back.processed_bubbles_info)
    for item in sorted_bubble_data:
        if item.get("is_outside_text"):
            continue
        a, b = live_info.get(tuple(item["bbox"])), back_info.get(tuple(item["bbox"]))
        if (a is None) != (b is None) or (a is not None and b is not None and not _equal(a, b)):
            raise RuntimeError(f"{page.id}: cleaning of {item['bbox']} doesn't round-trip")


# The region keys typesetting reads (typeset.page, typeset.narration, typeset.sfx_labels,
# clean.outside_text's fills); test_project checks this list against the code.
READ_KEYS = (
    "bbox",
    "is_outside_text",
    "mt_skip",
    "ocr_text",
    "original_crop_pil",
    "pending_fill",
    "is_dark_text",
    "text_color_rgb",
    "needs_text_background",
    "tilt_deg",
    "tilt_conf",
    "mt_narration",
    "mt_text",
)
INFO_KEYS = ("color", "mask", "base_mask", "is_seg", "is_colored", "text_bbox", "text_color_bgr")


def _render_info(processed: list[dict]) -> dict:
    """typeset.page's lookup of cleaning results by box, with the fields it reads."""
    return {
        tuple(info["bbox"]): {k: info.get(k) for k in INFO_KEYS}
        for info in processed
        if "bbox" in info and "color" in info and "mask" in info
    }


def _equal(a: Any, b: Any) -> bool:
    """Equal in value and in the types that matter (arrays: dtype too; images: mode too; an int
    is not a float). NumPy scalars count as their Python values."""
    a = a.item() if isinstance(a, np.generic) else a
    b = b.item() if isinstance(b, np.generic) else b
    if isinstance(a, Image.Image) or isinstance(b, Image.Image):
        return isinstance(a, Image.Image) and isinstance(b, Image.Image) and _same(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and np.array_equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if dataclasses.is_dataclass(a) and type(a) is type(b):
        return all(_equal(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b and (isinstance(a, (int, np.integer)) == isinstance(b, (int, np.integer)) or a is None)
