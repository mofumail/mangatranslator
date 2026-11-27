"""The pipeline's stages, each callable on its own, and run(): the batch pipeline as their
composition.

A WorkPage is a page being worked on: the live images and the stages' working data (the region
dicts the stage internals pass around, with a stable `region_id` each from detection on). The
project file stores it as a page document (mangatl.project.capture).

    detect(w)      bubbles, panels, outside text (without overlap its inpainting runs here too)
    clean(w)       bubble cleaning (with overlap: after the deferred outside-text inpainting)
    prepare(w)     the region crops for OCR and the LLM, reading order
    ocr(w)         read the ordered regions
    translate(w)   translation (one request per page, with story context and glossary); reads
                   the regions first when ocr() hasn't
    typeset(w)     the page's font sizes from its lettering
    render(w)      the translations drawn on the cleaned page (repeatable: works on copies)
"""

import base64
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from PIL import Image

from mangatl.clean.bubbles import clean_speech_bubbles, inpaints_colored_bubbles
from mangatl.clean.outside_text import finish_outside_text_work, paint_pending_osb_fills, process_outside_text
from mangatl.clean.page_glyph_map import PageGlyphMap
from mangatl.common.cancellation import CancellationManager, raise_if_manager_cancelled
from mangatl.common.exceptions import CancellationError, CleaningError, ImageProcessingError
from mangatl.common.image_utils import (
    cv2_to_pil,
    pil_to_cv2,
    resize_to_max_side,
    upscale_image_to_dimension,
    upscaler_for,
)
from mangatl.common.log import log_message
from mangatl.common.scaling import scale_length
from mangatl.detect.bubbles import detect_panels, detect_speech_bubbles
from mangatl.detect.outside_text import prepare_outside_text_work
from mangatl.detect.reading_order import sort_bubbles_by_reading_order
from mangatl.settings import Settings
from mangatl.translate.page import attach_ocr_texts, translate_or_error_texts, translate_page
from mangatl.translate.service import prepare_bubble_images_for_translation
from mangatl.typeset.page import render_translations, resolve_main_max_font, scaled_render_sizes


def _should_overlap_llm_with_inpaint(config: Settings) -> bool:
    return config.runtime.overlap_llm_with_inpaint and not config.runtime.cleaning_only


def _clean_speech_bubbles_for_page(
    pil_image_processed: Image.Image,
    bubble_data: list[dict[str, Any]],
    config: Settings,
    processing_scale: float,
    verbose: bool,
    fallback_cv_image: np.ndarray,
    glyph_map: PageGlyphMap | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Clean bubbles (colored bubbles are inpainted). Returns (cv_image, info)."""
    try:
        use_otsu = config.cleaning.otsu
        if inpaints_colored_bubbles(config):
            log_message(
                "Inpainting enabled for colored bubbles",
                verbose=verbose,
            )

        cleaned_image_cv, processed_bubbles_info = clean_speech_bubbles(
            pil_image_processed,
            config.models,
            config.detection.confidence,
            pre_computed_detections=bubble_data,
            thresholding_value=config.cleaning.threshold,
            use_otsu_threshold=use_otsu,
            auto_threshold=config.cleaning.auto_threshold,
            roi_shrink_px=config.cleaning.roi_shrink_px,
            verbose=verbose,
            processing_scale=processing_scale,
            conjoined_confidence=config.detection.conjoined_confidence,
            inpaint_colored_bubbles=inpaints_colored_bubbles(config),
            seed=config.cleaning.seed,
            osb_text_verification=config.detection.outside_text.cover_with_bubbles,
            inpaint_method=config.cleaning.inpainter,
            luminance_correction=config.cleaning.luminance_correction,
            bubble_detector_model=config.detection.bubble_model,
            request_coordinator=config.request_coordinator,
            glyph_map=glyph_map,
        )
        return cleaned_image_cv, processed_bubbles_info
    except CleaningError as e:
        log_message(f"Cleaning failed: {e}", always_print=True)
        return fallback_cv_image.copy(), []
    except CancellationError:
        raise
    except Exception as e:
        log_message(f"Error during cleaning: {e}", always_print=True)
        return fallback_cv_image.copy(), []


def get_image_encoding_params(pil_image_format: str | None) -> tuple[str, str]:
    """Returns (mime_type, cv2_ext) for a given PIL image format."""
    if pil_image_format and pil_image_format.upper() == "PNG":
        return "image/png", ".png"
    return "image/jpeg", ".jpg"


def _encode_context_image(image: Image.Image, config: Settings, cv2_ext: str) -> str:
    """Resize a page for LLM context per the context upscale setting, base64-encoded."""
    verbose = config.verbose
    # processing_scale is intentionally not used for context_image_max_side_pixels
    max_side = scale_length(
        config.translation.context_image_max_side,
        None,
        minimum=512,
        maximum=4096,
    )
    upscale_method = config.ocr.upscale

    upscaler = upscaler_for(config.models, upscale_method)
    if upscaler is not None:
        image = upscale_image_to_dimension(upscaler, image, max_side, "max", upscale_method, verbose)
        image = resize_to_max_side(image, max_side, verbose=verbose)
    elif upscale_method == "lanczos":
        image = resize_to_max_side(image, max_side, verbose=verbose, antiring=True)

    is_success, buffer = cv2.imencode(cv2_ext, pil_to_cv2(image))
    if not is_success:
        raise ImageProcessingError(f"Context image encoding to {cv2_ext} failed")
    return base64.b64encode(buffer).decode("utf-8")


def as_mode(image: Image.Image, mode: str) -> Image.Image:
    return image if image.mode == mode else image.convert(mode)


def _detect_page_bubbles(
    image_path: Path, image: Image.Image, config: Settings
) -> tuple[list[dict[str, Any]], list]:
    """Returns (bubbles, text-free boxes), both empty if detection fails."""
    log_message("Detecting speech bubbles...", verbose=config.verbose)
    try:
        return detect_speech_bubbles(
            image_path,
            config.models,
            config.detection.confidence,
            verbose=config.verbose,
            conjoined_detection=config.detection.conjoined,
            conjoined_confidence=config.detection.conjoined_confidence,
            image_override=image,
            osb_enabled=config.detection.outside_text.enabled,
            osb_text_verification=config.detection.outside_text.cover_with_bubbles,
            bubble_detector_model=config.detection.bubble_model,
        )
    except Exception as e:
        log_message(f"Error during detection: {e}", always_print=True)
        return [], []


def _detect_page_panels(image_path: Path, image: Image.Image, config: Settings) -> list | None:
    """Panels for reading order, or None when panel sorting is off or detection fails."""
    if not config.detection.panel_sorting:
        return None

    verbose = config.verbose
    try:
        log_message("Detecting panels...", verbose=verbose)
        panels = detect_panels(
            image_path,
            config.models,
            confidence=config.detection.panel_confidence,
            verbose=verbose,
            image_override=image,
        )
        if panels:
            log_message(f"Detected {len(panels)} panels", always_print=True)
        else:
            log_message("No panels detected", verbose=verbose)
    except Exception as e:
        log_message(f"Panel detection failed: {e}. Using global sorting.", always_print=True)
        panels = None
    return panels


def _encode_full_page_context(image: Image.Image, config: Settings, cv2_ext: str) -> str | None:
    try:
        encoded = _encode_context_image(image, config, cv2_ext)
    except Exception as e:
        log_message(f"Warning: Failed to encode full image context: {e}", always_print=True)
        return None
    log_message("Encoded full image for context", verbose=config.verbose)
    return encoded


def _clean_page(
    image: Image.Image,
    bubble_data: list[dict[str, Any]],
    config: Settings,
    processing_scale: float,
    target_mode: str,
    glyph_map: PageGlyphMap | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Returns (page with bubbles cleaned, in target_mode, per-bubble cleaning info).

    `glyph_map` is the text map of the page before outside text was inpainted, which
    still holds inside the bubbles.
    """
    if not bubble_data:
        return as_mode(image, target_mode), []
    log_message("Cleaning speech bubbles...", verbose=config.verbose)
    cleaned_cv, processed_bubbles_info = _clean_speech_bubbles_for_page(
        image,
        bubble_data,
        config,
        processing_scale,
        config.verbose,
        pil_to_cv2(image),
        glyph_map,
    )
    return as_mode(cv2_to_pil(cleaned_cv), target_mode), processed_bubbles_info


def _finish_deferred_cleaning(
    outside_work,
    image: Image.Image,
    bubble_data: list[dict[str, Any]],
    config: Settings,
    processing_scale: float,
    target_mode: str,
    glyph_map: PageGlyphMap | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Finish the deferred outside text inpainting, then clean the bubbles on top of it."""
    if outside_work is not None:
        image, _ = finish_outside_text_work(outside_work)
    return _clean_page(image, bubble_data, config, processing_scale, target_mode, glyph_map)


def _prepare_bubble_crops(
    bubble_data: list[dict[str, Any]],
    processed_bubbles_info: list[dict[str, Any]],
    page_cv: np.ndarray,
    config: Settings,
    mime_type: str,
) -> list[dict[str, Any]]:
    """Attach the refined cleaning masks to the bubbles and encode their crops for the LLM."""
    verbose = config.verbose
    log_message("Preparing bubble images...", verbose=verbose)

    mask_by_bbox: dict[tuple, Any] = {}
    for info in processed_bubbles_info:
        bbox_key = tuple(round(v) for v in info.get("bbox", ()))
        if len(bbox_key) != 4:
            continue
        mask = info.get("mask")
        if mask is None:
            mask = info.get("base_mask")
        if mask is not None:
            mask_by_bbox[bbox_key] = mask
    for bubble in bubble_data:
        bbox_key = tuple(round(v) for v in bubble.get("bbox", ()))
        if bbox_key in mask_by_bbox:
            bubble["seg_mask"] = mask_by_bbox[bbox_key]

    upscale_method = config.ocr.upscale

    bubble_data = prepare_bubble_images_for_translation(
        bubble_data,
        page_cv,
        upscaler_for(config.models, upscale_method),
        mime_type,
        config.ocr.bubble_min_side,
        upscale_method,
        config.ocr.whiteout_neighbors,
        verbose,
    )

    action = "Upscaled" if upscale_method != "none" else "Prepared"
    log_message(
        f"{action} {len(bubble_data)} bubble images for translation",
        always_print=True,
    )
    return bubble_data


def _sort_text_elements(
    valid_bubble_data: list[dict[str, Any]],
    outside_text_data: list[dict[str, Any]],
    config: Settings,
    panels: list | None,
) -> list[dict[str, Any]]:
    """Sort speech bubbles and outside text together into reading order."""
    reading_direction = config.translation.reading_direction
    if outside_text_data:
        log_message(
            f"Including {len(outside_text_data)} outside text regions in reading order calculation",
            verbose=config.verbose,
        )
    log_message(
        f"Sorting all text elements ({reading_direction.upper()})",
        verbose=config.verbose,
    )
    return sort_bubbles_by_reading_order(
        valid_bubble_data + outside_text_data, reading_direction, panels=panels
    )


# --- the working page ----------------------------------------------------------------------------
@dataclass
class WorkPage:
    config: Settings
    image: Image.Image  # the work image (without overlap: with outside text inpainted at detection)
    processing_scale: float
    target_mode: str
    mime_type: str
    cv2_ext: str
    image_format: str | None
    source_path: Path
    context_image: Image.Image  # the page as sent to the LLM for context
    cancel: CancellationManager | None = None
    # detect
    bubbles: list[dict[str, Any]] = field(default_factory=list)
    outside: list[dict[str, Any]] = field(default_factory=list)  # in detection order
    text_free_boxes: list = field(default_factory=list)
    panels: list | None = None
    glyph_map: PageGlyphMap | None = None
    outside_work: Any = None  # outside-text inpainting still to do (overlap)
    page_cv: np.ndarray | None = None
    full_image_b64: str | None = None
    next_region: int = 1
    locked: set[str] = field(default_factory=set)  # region ids the pipeline leaves as they are
    edited: set[str] = field(default_factory=set)  # region ids carrying a person's work
    # clean
    canvas: Image.Image | None = None  # the page text is drawn on
    cleaning: list[dict[str, Any]] = field(default_factory=list)  # per cleaned bubble
    # prepare / translate
    ordered: list[dict[str, Any]] = field(default_factory=list)  # regions in reading order
    translations: list[str] = field(default_factory=list)  # aligned with `ordered`
    ocr_texts: list[str] = field(default_factory=list)
    # typeset
    sizes: Any = None
    glyphs: dict = field(default_factory=dict)  # the page's lettering sizes (auto font size)
    typeset_done: bool = False

    @property
    def overlap(self) -> bool:
        """Outside-text inpainting and bubble cleaning run alongside the LLM call."""
        return _should_overlap_llm_with_inpaint(self.config)

    def check_cancelled(self) -> None:
        raise_if_manager_cancelled(self.cancel)

    def new_ids(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            item["region_id"] = f"r{self.next_region}"
            self.next_region += 1


# --- stages ---------------------------------------------------------------------------------------
def detect(w: WorkPage) -> WorkPage:
    """Speech bubbles, panels and outside text. Without overlap, outside text is also inpainted
    here (into w.image)."""
    config, verbose = w.config, w.config.verbose
    # Speech bubbles first, so outside-text processing can respect bubble regions
    w.bubbles, w.text_free_boxes = _detect_page_bubbles(w.source_path, w.image, config)
    w.check_cancelled()
    w.panels = _detect_page_panels(w.source_path, w.image, config)
    w.check_cancelled()
    # Built on first use from the original page, then shared by OSB and bubble cleaning
    w.glyph_map = PageGlyphMap.for_page(w.image, config, verbose)
    if w.overlap:
        w.outside_work = prepare_outside_text_work(
            w.image,
            config,
            w.source_path,
            w.image_format,
            verbose,
            bubble_data=w.bubbles,
            text_free_boxes=w.text_free_boxes,
            panels=w.panels,
            glyph_map=w.glyph_map,
        )
        w.outside = w.outside_work.outside_text_data if w.outside_work is not None else []
    else:
        w.image, w.outside = process_outside_text(
            w.image,
            config,
            w.source_path,
            w.image_format,
            verbose,
            bubble_data=w.bubbles,
            text_free_boxes=w.text_free_boxes,
            panels=w.panels,
            glyph_map=w.glyph_map,
        )
    w.new_ids(w.bubbles)
    w.new_ids(w.outside)
    # With overlap this is the pre-inpaint page, which the bubble and OSB LLM crops use
    w.page_cv = pil_to_cv2(w.image)
    if config.translation.full_page_context:
        w.full_image_b64 = _encode_full_page_context(w.context_image, config, w.cv2_ext)
    w.check_cancelled()
    return w


def clean(w: WorkPage, region_ids: Iterable[str] | None = None) -> WorkPage:
    """Clean the speech bubbles (after the deferred outside-text inpainting, with overlap). With
    `region_ids` and a cleaned page, only those bubbles are cleaned, on top of it (outside text is
    left as it is)."""
    if region_ids is not None and w.canvas is not None:
        scope = in_scope(w, region_ids)
        picked = [b for b in w.bubbles if b["region_id"] in scope]
        boxes = {tuple(b["bbox"]) for b in picked}
        canvas, infos = (
            _clean_page(w.canvas, picked, w.config, w.processing_scale, w.target_mode, w.glyph_map)
            if picked
            else (w.canvas, [])
        )
        w.canvas = canvas
        w.cleaning = [c for c in w.cleaning if tuple(c.get("bbox", ())) not in boxes] + infos
        return w
    w.canvas, w.cleaning = _finish_deferred_cleaning(
        w.outside_work if w.overlap else None,
        w.image,
        w.bubbles,
        w.config,
        w.processing_scale,
        w.target_mode,
        w.glyph_map,
    )
    return w


def _box(item: dict) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in item["bbox"])
    return x1, y1, x2, y2


class Redetected(NamedTuple):
    same: list[str]  # found again in place: reads and translations still hold
    moved: list[str]  # found again, but its box changed: to be read and translated again
    added: list[str]  # new regions
    removed: list[str]  # no longer found
    kept: list[str]  # no longer found, but kept (flagged "vanished") for a person's work or a lock


def redetect(w: WorkPage, area: tuple[int, int, int, int] | None = None) -> Redetected:
    """Detect again (the whole page, or the regions mostly inside `area`), keeping the id of every
    region the new detection finds again (mangatl.project.match). Locked regions keep their
    detection. Then clean, prepare and read / translate what changed (`moved` + `added`).

    The whole page: everything is cleaned again (clean(w)). An area: clean(w, ids) cleans its
    bubbles on the cleaned page; outside text in it is not inpainted again."""
    from mangatl.project.match import SAME_IOU, match, overlaps

    fresh = WorkPage(
        config=w.config,
        image=w.context_image.copy(),
        processing_scale=w.processing_scale,
        target_mode=w.target_mode,
        mime_type=w.mime_type,
        cv2_ext=w.cv2_ext,
        image_format=w.image_format,
        source_path=w.source_path,
        context_image=w.context_image,
        cancel=w.cancel,
    )
    detect(fresh)

    def kind(item: dict) -> str:
        return "outside" if item.get("is_outside_text") else "bubble"

    def inside(item: dict) -> bool:
        return area is None or overlaps(_box(item), area)

    old = [item for item in [*w.bubbles, *w.outside] if inside(item)]
    new = [item for item in [*fresh.bubbles, *fresh.outside] if inside(item)]
    m = match(
        [(i["region_id"], kind(i), _box(i)) for i in old],
        [(kind(i), _box(i)) for i in new],
    )
    old_by_id = {item["region_id"]: item for item in old}
    result = Redetected([], [], [], [], [])
    take: dict[int, dict] = {}  # new index -> the item standing for it
    for ni, (rid, score) in m.matched.items():
        if rid in w.locked:
            take[ni] = old_by_id[rid]
            continue
        new[ni]["region_id"] = rid
        take[ni] = new[ni]
        (result.same if score >= SAME_IOU else result.moved).append(rid)
    for ni in m.added:
        w.new_ids([new[ni]])
        take[ni] = new[ni]
        result.added.append(new[ni]["region_id"])
    keep_old = []
    for rid in m.vanished:
        if rid in w.locked or rid in w.edited:
            old_by_id[rid]["region_flag"] = "vanished"
            keep_old.append(old_by_id[rid])
            result.kept.append(rid)
        else:
            result.removed.append(rid)
    # reads and translations no longer hold for moved regions (prepare() realigns the rest by id)
    for i, item in enumerate(w.ordered):
        if item["region_id"] in result.moved and i < len(w.translations) and i < len(w.ocr_texts):
            w.ocr_texts[i], w.translations[i] = "", ""
            item.pop("ocr_text", None)

    gone = set(result.removed)
    if area is None:  # the new detection is the page's
        items = [take[ni] for ni in range(len(new))] + keep_old
        w.image, w.outside_work, w.glyph_map = fresh.image, fresh.outside_work, fresh.glyph_map
        w.panels, w.text_free_boxes, w.page_cv = fresh.panels, fresh.text_free_boxes, fresh.page_cv
        w.full_image_b64 = fresh.full_image_b64
        w.canvas, w.cleaning = None, []
    else:  # the rest of the page stays as it was
        replaced = {item["region_id"]: item for item in take.values()}
        outside_area = [i for i in [*w.bubbles, *w.outside] if not inside(i)]
        kept_ids = {item["region_id"] for item in keep_old}
        inside_kept = [  # the area's regions found again (or kept), in their old places
            replaced.get(i["region_id"], i)
            for i in old
            if i["region_id"] in replaced or i["region_id"] in kept_ids
        ]
        added = [take[ni] for ni in m.added]
        items = outside_area + inside_kept + added
        w.outside_work, w.glyph_map = None, fresh.glyph_map  # the glyph map cleaning uses
    w.bubbles = [i for i in items if kind(i) == "bubble" and i["region_id"] not in gone]
    w.outside = [i for i in items if kind(i) == "outside" and i["region_id"] not in gone]
    return result


def prepare(w: WorkPage, refine_masks: bool = False) -> WorkPage:
    """The bubble crops for OCR and the LLM, then all regions with something to read, in reading
    order. `refine_masks`: first replace each bubble's detection mask by its cleaning mask (the
    batch pipeline, when cleaning ran before; a stored page keeps the masks its crops came from)."""
    assert w.page_cv is not None
    # what reading and translating found so far, by region: a page prepared again (re-detected)
    # keeps it for the regions found again
    known: dict[str, tuple[str, str, dict]] = {}
    if len(w.ocr_texts) == len(w.ordered) and len(w.translations) == len(w.ordered):
        known = {
            item["region_id"]: (o, t, {k: item[k] for k in TEXT_STAGE_KEYS if k in item})
            for item, o, t in zip(w.ordered, w.ocr_texts, w.translations)
        }
    w.bubbles = _prepare_bubble_crops(
        w.bubbles, w.cleaning if refine_masks else [], w.page_cv, w.config, w.mime_type
    )
    valid = [b for b in w.bubbles if b.get("image_b64")]
    w.ordered = _sort_text_elements(valid, w.outside, w.config, w.panels) if valid or w.outside else []
    if known:
        empty = ("", "", {})
        for item in w.ordered:
            item.update(known.get(item["region_id"], empty)[2])
        w.ocr_texts = [known.get(item["region_id"], empty)[0] for item in w.ordered]
        w.translations = [known.get(item["region_id"], empty)[1] for item in w.ordered]
    return w


# Region fields set by reading and translating (OCR: the read and the other-language skip;
# translation: narration on the art)
TEXT_STAGE_KEYS = ("ocr_text", "mt_skip", "mt_narration", "mt_text")


def _crops(w: WorkPage) -> tuple[list[str], list[str]]:
    images = [b["image_b64"] for b in w.ordered if "image_b64" in b]
    mimes = [b["mime_type"] for b in w.ordered if "image_b64" in b and "mime_type" in b]
    return images, mimes


def in_scope(w: WorkPage, region_ids: Iterable[str] | None) -> set[str]:
    """The regions a stage call works on: those asked for (None: all), never a locked one."""
    every = {item["region_id"] for item in [*w.bubbles, *w.outside, *w.ordered]}
    return (every if region_ids is None else set(region_ids) & every) - w.locked


def ocr(w: WorkPage, region_ids: Iterable[str] | None = None) -> WorkPage:
    """Read the ordered regions (all, or the given ones) and set their ocr_text."""
    from mangatl.ocr.page import run_ocr  # heavy (OCR models)

    scope = in_scope(w, region_ids)
    picked = [i for i, item in enumerate(w.ordered) if item["region_id"] in scope and item.get("image_b64")]
    reads = (
        run_ocr(
            w.config,
            [w.ordered[i]["image_b64"] for i in picked],
            [w.ordered[i] for i in picked],
            w.config.verbose,
        )
        if picked
        else []
    )
    texts = (
        list(w.ocr_texts)
        if len(w.ocr_texts) == len(w.ordered)
        else [item.get("ocr_text") or "" for item in w.ordered]
    )
    for i, text in zip(picked, reads, strict=True):
        texts[i] = text
    w.ocr_texts = texts
    attach_ocr_texts(w.ordered, w.ocr_texts, w.config.verbose)
    return w


def translate(
    w: WorkPage,
    previous_context_texts: list[list[str]] | None = None,
    previous_context_texts_provider: Callable[[], list[list[str]]] | None = None,
    ocr_texts_out: list[str] | None = None,
    alongside: Callable[[], Any] | None = None,
    region_ids: Iterable[str] | None = None,
) -> WorkPage:
    """Translate the ordered regions (one request for the page, with the previous pages' text and
    the glossary), reading them first unless ocr() has. `alongside` runs at the same time
    (cleaning, with overlap). With `region_ids`, the request is the same (the page's context)
    but only those regions' translations change."""
    scope = in_scope(w, region_ids)
    before = list(w.translations) if len(w.translations) == len(w.ordered) else None
    images, mimes = _crops(w)
    if not images:
        log_message("No valid bubbles after sorting", always_print=True)
        if alongside is not None:
            alongside()
        w.translations, w.ocr_texts = [], []
        return w
    call_api = partial(
        translate_page,
        w.config,
        w.ordered,
        images,
        mimes,
        w.full_image_b64,
        w.mime_type,
        previous_context_texts or [],
        previous_context_texts_provider,
        reads=list(w.ocr_texts) if len(w.ocr_texts) == len(images) else None,
    )
    finish = partial(
        translate_or_error_texts,
        sorted_bubble_data=w.ordered,
        image_count=len(images),
        config=w.config,
        ocr_texts_out=ocr_texts_out,
    )
    if alongside is not None:
        log_message("Running LLM translation concurrently with inpainting", always_print=True)
        with ThreadPoolExecutor(max_workers=2) as executor:
            side = executor.submit(alongside)
            translation = executor.submit(call_api)
            try:
                side.result()
            except Exception:
                translation.cancel()
                raise
            w.translations, w.ocr_texts = finish(translation.result)
    else:
        w.translations, w.ocr_texts = finish(call_api)
    if before is not None and len(w.translations) == len(before):
        w.translations = [
            new if item["region_id"] in scope else old
            for item, old, new in zip(w.ordered, before, w.translations, strict=True)
        ]
    if w.ocr_texts:
        attach_ocr_texts(w.ordered, w.ocr_texts, w.config.verbose)
    return w


def typeset(w: WorkPage, strip_glyph_size: list[float] | None = None) -> WorkPage:
    """The page's font sizes: scaled to the page, the dialogue size matched to its lettering."""
    assert w.page_cv is not None
    sizes = scaled_render_sizes(w.config, w.processing_scale)
    w.glyphs = {}
    w.sizes = sizes._replace(
        main_max_font=resolve_main_max_font(
            w.config,
            w.page_cv,
            w.cleaning,
            sizes.main_min_font,
            sizes.main_max_font,
            w.processing_scale,
            strip_glyph_size,
            w.config.verbose,
            w.glyphs,
        )
    )
    w.typeset_done = True
    return w


def render(w: WorkPage, empty_bubbles: list | None = None, outcomes: dict | None = None) -> Image.Image:
    """The translations drawn on the cleaned page, with the outside-text fills. The page and
    its regions are left as they are (rendering works on copies), so a page renders again the
    same way. `empty_bubbles` receives the boxes of bubbles left without text; `outcomes`, per
    region id, how its text came out: {"outcome": "fit" | "below_min_font" | "kept_original_no_fit"
    | "kept_original" | "empty" | "failed", "font_size", "max_font"}."""
    assert w.canvas is not None and w.page_cv is not None
    copies = {id(item): dict(item) for item in [*w.ordered, *w.outside]}
    image = render_translations(
        w.canvas.copy(),
        [copies[id(item)] for item in w.ordered],
        list(w.translations),
        w.cleaning,
        w.page_cv,
        w.config,
        w.sizes,
        w.processing_scale,
        empty_bubbles if empty_bubbles is not None else [],
        w.cancel,
        w.glyphs,
    )
    if outcomes is not None:
        for item in w.ordered:
            done = copies[id(item)]
            outcomes[item["region_id"]] = {
                "outcome": done.get("mt_outcome", "empty"),
                "font_size": done.get("mt_font_size"),
                "max_font": done.get("mt_max_font"),
            }
    return paint_pending_osb_fills(image, [copies[id(item)] for item in w.outside])


def result(w: WorkPage) -> Image.Image:
    """The page as it stands: rendered when typeset, else the cleaned (or work) page with the
    outside-text fills."""
    if w.typeset_done:
        return render(w)
    if w.canvas is None:
        return w.image
    return paint_pending_osb_fills(w.canvas.copy(), [dict(item) for item in w.outside])


def run(
    w: WorkPage,
    previous_context_texts: list[list[str]] | None = None,
    previous_context_texts_provider: Callable[[], list[list[str]]] | None = None,
    ocr_texts_out: list[str] | None = None,
    strip_glyph_size: list[float] | None = None,
    empty_bubbles: list | None = None,
) -> Image.Image:
    """The whole pipeline on one page; returns the translated page."""
    detect(w)
    if not w.bubbles and not w.outside:
        log_message("No speech bubbles or outside text detected", always_print=True)
        return w.image
    if w.bubbles:
        log_message(f"Detected {len(w.bubbles)} bubbles", verbose=w.config.verbose)
    if w.outside:
        log_message(f"Detected {len(w.outside)} outside text regions", verbose=w.config.verbose)

    if w.overlap:  # cleaning waits for the LLM call, to run beside it
        w.canvas, w.cleaning = as_mode(w.image, w.target_mode), []
    else:
        clean(w)
    if w.config.runtime.cleaning_only:
        log_message("Cleaning only mode - skipping translation", always_print=True)
        return result(w)

    prepare(w, refine_masks=not w.overlap)  # with overlap, cleaning hasn't run yet
    if not w.ordered:
        log_message("No valid bubble images or outside text for translation", always_print=True)
        if w.overlap:
            clean(w)
        return result(w)
    w.check_cancelled()
    translate(
        w,
        previous_context_texts,
        previous_context_texts_provider,
        ocr_texts_out,
        alongside=partial(clean, w) if w.overlap else None,
    )
    typeset(w, strip_glyph_size)
    return render(w, empty_bubbles)
