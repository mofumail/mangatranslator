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


NARRATION_MIN_WORDS = 6  # outside-bubble translations this long are narration, not SFX


def utterance_lines(text: str) -> str:
    """One line per quoted utterance / narration sentence run, like the source's columns."""
    t = re.sub(r'(["”’\'])\s+(?=["“‘\'])', r"\1\n", text)       # "…" "…"
    t = re.sub(r'([.!?~…♡])\s+(?=["“‘]|\'[A-Z])', r"\1\n", t)   # narration. "…" / 'Comment
    t = re.sub(r'(["”])\s+(?=[A-Z])', r"\1\n", t)               # "…" Narration
    return t


def install_narration_style() -> None:
    """Upstream sets every outside-bubble text like a sound effect: upper case, display font
    (Bangers), following the region's tilt. Long ones are narration/dialogue, which reads far
    better in the dialogue font, normal case, upright, one utterance per line."""
    import dataclasses

    import core.pipeline as pipeline

    original_batch = pipeline.call_translation_api_batch

    def call_translation_api_batch(*args, **kwargs):
        texts = original_batch(*args, **kwargs)
        for i, m in enumerate(kwargs.get("bubble_metadata") or []):
            if i < len(texts) and m.get("is_outside_text") and isinstance(texts[i], str):
                t = texts[i].strip()
                if len(t.split()) >= NARRATION_MIN_WORDS and not t.startswith("["):
                    m["mt_narration"] = True
                    m["mt_text"] = utterance_lines(t)
        return texts

    pipeline.call_translation_api_batch = call_translation_api_batch

    original_attempts = pipeline._osb_render_attempts

    def _osb_render_attempts(bubble, config, sizes):
        if bubble.get("mt_narration"):
            return [(0.0, False, None), (0.0, True, None)]  # upright
        return original_attempts(bubble, config, sizes)

    pipeline._osb_render_attempts = _osb_render_attempts

    original_render = pipeline._render_osb_attempt

    def _render_osb_attempt(image, bubble, text, bubble_id, config, sizes, attempt, layout_only=False):
        if bubble.get("mt_narration"):
            text = bubble.get("mt_text", text)  # upstream passes text.upper()
            config = dataclasses.replace(
                config, outside_text=dataclasses.replace(config.outside_text, osb_font_dir=None))
        return original_render(image, bubble, text, bubble_id, config, sizes, attempt, layout_only)

    pipeline._render_osb_attempt = _render_osb_attempt


def install_keep_line_breaks() -> None:
    """Upstream's renderer collapses all whitespace (newlines included) right before layout,
    although its layout engine honors explicit line breaks. Rebuild render_text_skia from its
    own source with that one line changed to keep newlines. If upstream changes the line, the
    patch is skipped (narration then wraps as one paragraph, as before)."""
    import inspect
    import textwrap

    import core
    import core.pipeline as pipeline
    import core.text as text_pkg
    import core.text.placeholders as placeholders
    from core.text import text_renderer
    from utils.logging import log_message

    old = 'clean_text = " ".join(normalized_text.split())'
    new = 'clean_text = "\\n".join(" ".join(l.split()) for l in normalized_text.split("\\n")).strip()'
    src = textwrap.dedent(inspect.getsource(text_renderer.render_text_skia))
    if src.count(old) != 1:
        log_message("Note: line-break patch for the text renderer no longer applies; skipped",
                    always_print=True)
        return
    namespace = text_renderer.__dict__
    exec(compile(src.replace(old, new), text_renderer.__file__, "exec"), namespace)
    patched = namespace["render_text_skia"]
    for module in (text_renderer, pipeline, placeholders, text_pkg, core):
        module.render_text_skia = patched


def install_wrapped_paragraphs() -> None:
    """Upstream's layout keeps explicit newlines but never wraps inside those lines, so long
    utterances shrink the whole block. Wrap each paragraph with upstream's own line breaker
    at the same font size, then stack them."""
    from core.text import layout_engine as le

    original = le.check_fit

    def check_fit(font_size, text, max_render_width, max_render_height, *args, **kwargs):
        if "\n" not in text or kwargs.get("vertical_stack"):
            return original(font_size, text, max_render_width, max_render_height, *args, **kwargs)
        lines, metrics, line_height, widest = [], None, None, 0.0
        for para in (p for p in text.split("\n") if p.strip()):
            fit = original(font_size, para, max_render_width, float("inf"), *args, **kwargs)
            if fit is None:
                return None  # a word doesn't fit the width at this size
            lines += fit["lines"]
            metrics, line_height = fit["metrics"], fit["line_height"]
            widest = max(widest, fit["max_line_width"])
        if not lines:
            return None
        block_height = (-metrics.fAscent + metrics.fDescent) + (len(lines) - 1) * line_height
        if block_height > max_render_height:
            return None
        return {"lines": lines, "metrics": metrics, "max_line_width": widest,
                "line_height": line_height, "block_height": block_height}

    le.check_fit = check_fit


def _content_len(text: str) -> int:
    return sum(1 for c in text if c.isalnum())


def install_truncation_guard() -> None:
    """The LLM occasionally returns only the first sentence of a long block (seen: a 120-char
    narration block came back as "-A little while later."), silently dropping the dialogue.
    If a long source line gets an implausibly short translation, translate the page once more
    (bypassing upstream's cache) and keep, per line, the fuller of the two answers."""
    import core.pipeline as pipeline
    from core.caching import get_cache
    from utils.logging import log_message

    original_batch = pipeline.call_translation_api_batch

    def looks_truncated(src: str, tr: str) -> bool:
        n = _content_len(src)
        return n >= 30 and isinstance(tr, str) and _content_len(tr) < 0.5 * n

    def call_translation_api_batch(*args, **kwargs):
        texts = original_batch(*args, **kwargs)
        ocr = kwargs.get("ocr_texts_output") or []
        sources = [m.get("ocr_text") or (ocr[i] if i < len(ocr) else "")
                   for i, m in enumerate(kwargs.get("bubble_metadata") or [])]
        bad = [i for i, (s, t) in enumerate(zip(sources, texts)) if looks_truncated(s, t)]
        if not bad:
            return texts
        log_message(f"Translation looks truncated for item(s) {[i + 1 for i in bad]}; retrying once",
                    always_print=True)
        get_cache().clear_translation_cache()
        retry_kwargs = dict(kwargs)
        if "ocr_texts_output" in retry_kwargs:
            retry_kwargs["ocr_texts_output"] = []
        again = original_batch(*args, **retry_kwargs)
        for i in bad:
            if i < len(again) and _content_len(again[i]) > _content_len(texts[i]):
                texts[i] = again[i]
        return texts

    pipeline.call_translation_api_batch = call_translation_api_batch
