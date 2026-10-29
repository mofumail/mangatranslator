"""OCR for text outside speech bubbles (narration, captions, chat lines), installed into
MangaTranslator at runtime by mt_upstream.py; the submodule itself stays untouched.

Upstream reads every text region with manga-ocr in one pass. manga-ocr squashes its input to
224x224, so a narration block with many lines turns into a few pixels per glyph and it
"reads" fluent but invented text. Instead, the established design from manga-image-translator
and BallonsTranslator is used: detect the individual text lines (comic-text-detector), read
each line with a recognizer that keeps the aspect ratio (Hayai OCR), and join them in reading
order. Speech bubbles and single-line regions (e.g. sound effects) keep upstream's path.
"""

from __future__ import annotations

import itertools
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

MIN_LINES = 2  # regions with at least this many detected lines are read line by line


def _script_ratios(text: str) -> tuple[int, float, float]:
    chars = [c for c in text if c.isalnum()]
    if not chars:
        return 0, 0.0, 0.0
    kana = sum(1 for c in chars if "぀" <= c <= "ヿ")
    han = sum(1 for c in chars if "一" <= c <= "鿿" or "㐀" <= c <= "䶿")
    return len(chars), kana / len(chars), han / len(chars)


def is_japanese(text: str) -> bool:
    n, kana, _ = _script_ratios(text)
    return n >= 4 and kana >= 0.2


# Characters that are everyday Chinese but (almost) never appear in Japanese text:
# sentence particles and simplified-only forms.
ZH_MARKERS = set("们呢吧吗么这嗯哦啊呀您她说让没还对给吃过时间样边见")


def is_other_cjk(text: str) -> bool:
    """A block in Chinese, e.g. a translation printed next to the Japanese: little kana
    (OCR may misread a few glyphs as kana) plus Chinese-only particles, or no kana at all."""
    n, kana, han = _script_ratios(text)
    if n < 6 or han < 0.4:
        return False
    markers = sum(1 for c in text if c in ZH_MARKERS)
    return (kana < 0.08 and n >= 12) or (kana < 0.25 and markers >= 2)


# CTD is a page-level detector (trained on whole pages at 1024 px), so it runs once per page,
# like in manga-image-translator; each region then takes the lines that fall inside it.
# Pages are registered when upstream builds its outside-text regions (same coordinates).
_page_lock = threading.Lock()
_page_ids = itertools.count(1)
_PAGES: "OrderedDict[int, object]" = OrderedDict()
_PAGE_LINES: dict[int, list] = {}


def _register_page(pil_image) -> int:
    with _page_lock:
        token = next(_page_ids)
        _PAGES[token] = pil_image
        while len(_PAGES) > 4:  # a few pages in flight at most
            old, _ = _PAGES.popitem(last=False)
            _PAGE_LINES.pop(old, None)
        return token


def _page_lines(token: int, models_dir: Path) -> list:
    import text_lines

    with _page_lock:
        if token in _PAGE_LINES:
            return _PAGE_LINES[token]
        page = _PAGES.get(token)
    lines = text_lines.get(models_dir).detect(page) if page is not None else []
    with _page_lock:
        _PAGE_LINES[token] = lines
    return lines


def _lines_in(lines: list, box) -> list:
    """Lines lying mostly (>= 60% of their area) inside the region box."""
    x1, y1, x2, y2 = box
    out = []
    for l in lines:
        ix = max(0, min(x2, l.x2) - max(x1, l.x1))
        iy = max(0, min(y2, l.y2) - max(y1, l.y1))
        if ix * iy >= 0.6 * (l.x2 - l.x1) * (l.y2 - l.y1):
            out.append(l)
    return out


def read_region(meta: dict, models_dir: Path, rtl: bool = True) -> str | None:
    """Hayai OCR of one outside-text region: line by line when it holds MIN_LINES or more
    detected lines, otherwise the whole region in one pass (sound effects, short asides; Hayai
    was trained on onomatopoeia, manga-ocr was not). None (keep upstream's read) if the page
    is unknown."""
    import hayai_ocr
    import text_lines

    token = meta.get("mt_page")
    box = meta.get("original_bbox") or meta.get("bbox")
    page = _PAGES.get(token) if token else None
    if page is None or not box or len(box) != 4:
        return None
    lines = text_lines.reading_order(_lines_in(_page_lines(token, models_dir), box), rtl=rtl)
    W, H = page.size
    if len(lines) < MIN_LINES:
        x1, y1, x2, y2 = (int(v) for v in box)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        text = hayai_ocr.get(models_dir).read([page.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))],
                                             max_patches=512, max_new_tokens=64)[0].strip()
        return text or None
    pieces = []
    for line in lines:
        pad = max(2, int(0.1 * line.thickness))
        pieces.append(page.crop((max(0, line.x1 - pad), max(0, line.y1 - pad),
                                 min(W, line.x2 + pad), min(H, line.y2 + pad))))
    texts = [t.strip() for t in hayai_ocr.get(models_dir).read(pieces, max_patches=512, max_new_tokens=160)]
    texts = [t for t in texts if t]
    if not texts:
        return None
    cjk = sum(_script_ratios(t)[1] + _script_ratios(t)[2] for t in texts) / len(texts) > 0.5
    return ("" if cjk else " ").join(texts)


_SKIPPED_BBOXES: set[tuple] = set()  # blocks erased on purpose; not "left empty" errors


def install() -> None:
    import core.pipeline as pipeline
    from core.ml.model_manager import get_model_manager
    from core.services import translation as tr
    from utils.logging import log_message

    import core.outside_text_processor as osb

    original_build = osb._build_outside_text_data

    def _build_outside_text_data(pil_image, *args, **kwargs):
        items = original_build(pil_image, *args, **kwargs)
        if items:
            token = _register_page(pil_image)
            for item in items:
                item["mt_page"] = token
        return items

    osb._build_outside_text_data = _build_outside_text_data

    original = tr._perform_manga_ocr

    def perform_manga_ocr(images_b64, bubble_metadata, debug=False):
        texts = original(images_b64, bubble_metadata, debug)
        models_dir = get_model_manager().models_dir
        changed = False
        for i, meta in enumerate(bubble_metadata):
            if i >= len(texts) or not meta.get("is_outside_text"):
                continue
            text = read_region(meta, models_dir)
            if text:
                texts[i] = text
                changed = True

        # Pages carrying both the Japanese text and a Chinese translation of it: erase the
        # Chinese block without rendering anything, and keep it away from the translator.
        if not os.environ.get("MT_KEEP_OTHER_LANGUAGE") and any(is_japanese(t) for t in texts):
            for i, t in enumerate(texts):
                if i < len(bubble_metadata) and is_other_cjk(t):
                    bubble_metadata[i]["mt_skip"] = True
                    _SKIPPED_BBOXES.add(tuple(bubble_metadata[i].get("bbox", ())))
                    texts[i] = "(no text)"
                    changed = True
                    log_message(f"Skipping block {i + 1}: not in the source language "
                                "(erased, not translated)", always_print=True)
        if changed:
            log_message("Text outside bubbles re-read line by line (comic-text-detector + Hayai OCR):",
                        always_print=True)
            tr._format_ocr_results(texts, bubble_metadata)
        return texts

    tr._perform_manga_ocr = perform_manga_ocr

    original_batch = pipeline.call_translation_api_batch

    def call_translation_api_batch(*args, **kwargs):
        texts = original_batch(*args, **kwargs)
        for i, t in enumerate(texts):
            # Upstream fills items the LLM skipped with "[<provider>-Translate: missing item N]",
            # which its renderer doesn't recognize as an error and draws onto the page.
            # Mark them as failed instead, so the original lettering is restored.
            if isinstance(t, str) and re.fullmatch(r"\s*\[[^\]]*missing item \d+\]\s*", t, re.I):
                texts[i] = "[OCR FAILED]"
        for i, m in enumerate(kwargs.get("bubble_metadata") or []):
            if m.get("mt_skip") and i < len(texts):
                texts[i] = ""  # rendered as "erase only"
        return texts

    pipeline.call_translation_api_batch = call_translation_api_batch

    def _translate_batch_page(*args, **kwargs) -> None:
        # upstream fails a page if any region ends up without text; skipped blocks are intended
        empty: list = []
        pipeline.translate_and_render(*args, empty_bubbles_out=empty, **kwargs)
        real = [b for b in empty if tuple(b) not in _SKIPPED_BBOXES]
        if real:
            raise pipeline.TranslationError(f"{len(real)} bubble(s) left empty")

    pipeline._translate_batch_page = _translate_batch_page


FLAT_P90_MAX = 6  # paper measured 2-6; blurred dark art 9-10


def install_flat_fill_check() -> None:
    """Upstream paints a flat rectangle when ~95% of the pixels around a text region are
    within 15 levels of their median, snapping dark medians to pure black. Blurred dark art
    passes that test, so it got hard black boxes. Only treat a border as solid when it is truly
    flat (90% of pixels within FLAT_P90_MAX); everything else goes to the inpainting model."""
    import numpy as np

    import core.outside_text_processor as osb

    original = osb._border_color

    def _border_color(pixels):
        color, solid_ratio = original(pixels)
        median = np.median(pixels, axis=0).astype(int)
        p90 = float(np.percentile(np.max(np.abs(pixels.astype(int) - median), axis=1), 90))
        if p90 > FLAT_P90_MAX:
            solid_ratio = min(solid_ratio, osb.OSB_SOLID_RATIO_THRESHOLD - 0.01)
        return color, solid_ratio

    osb._border_color = _border_color


def install_no_downscale() -> None:
    """Upstream resizes every crop to a 128 px short side for cloud LLMs, shrinking big
    narration blocks to unreadable thumbnails. For local OCR only upscale, never downscale."""
    import core.image as image_pkg
    import core.outside_text_processor as osb
    from core.image import image_utils
    from core.services import translation as tr

    original = image_utils.process_bubble_image_cached

    def process_bubble_image_cached(bubble_image_pil, upscale_model, device,
                                    target_min_side=200, mode="min", *args, **kwargs):
        if mode == "min" and min(bubble_image_pil.size) >= target_min_side:
            return bubble_image_pil
        return original(bubble_image_pil, upscale_model, device, target_min_side, mode, *args, **kwargs)

    for module in (image_utils, image_pkg, osb, tr):
        module.process_bubble_image_cached = process_bubble_image_cached
