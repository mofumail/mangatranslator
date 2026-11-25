"""OCR for text outside speech bubbles (narration, captions, chat lines), installed into
the pipeline (mangatl/, derived from MangaTranslator) at runtime by mt_upstream.py.

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
# Simplified forms Japanese never uses (Japanese writes 這説話時過対開関門見譲還発気...)
ZH_SIMPLIFIED = set("们这说话时过对开关门见让还发气车东经书长问间听现实认识应该样边头动从觉为无么个给吗呢吧"
                    "爱乐欢够难变种进远连级员单层战视请谁论读课谢买卖脑脸点儿哪")


def is_other_cjk(text: str) -> bool:
    """A block in Chinese, e.g. a translation printed next to the Japanese: little kana
    (OCR may misread a few glyphs as kana) plus Chinese-only particles, or no kana at all."""
    n, kana, han = _script_ratios(text)
    if n < 6 or han < 0.4:
        return False
    markers = sum(1 for c in text if c in ZH_MARKERS)
    simplified = sum(1 for c in text if c in ZH_SIMPLIFIED or c in ZH_MARKERS)
    # No kana alone is not enough: Japanese status screens and labels are often all kanji
    # (状態：乳房開発完了 人格消失), so also ask for characters only Chinese writes this way.
    return (kana < 0.08 and n >= 8 and simplified >= 2) or (kana < 0.25 and markers >= 2)


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


def _pad_mean(im, frac: float = 0.25):
    """Margin in the crop's mean color. Hayai misreads tight crops of hand-drawn sound effects
    (むに -> むり, ん -> き); a margin fixes most. Not used for clean lines, where it can hurt."""
    from PIL import ImageOps

    fill = tuple(int(v) for v in im.resize((1, 1)).getpixel((0, 0)))
    return ImageOps.expand(im, int(max(im.size) * frac), fill=fill)


def pick_read(other: str, preferred: str) -> str:
    """Two OCR reads of one region: keep the longer when one contains the other (the usual
    failure is stopping early: ドクッ -> ド, 十年後ーー夏 -> 十年後), else the preferred one."""
    import unicodedata

    def letters(t: str) -> str:  # a read of only "..." must not count as contained
        t = unicodedata.normalize("NFKC", t or "")
        return "".join(c for c in t if unicodedata.category(c)[0] in "LN")

    a, b = letters(other), letters(preferred)
    if a and b and (a in b or b in a):
        if len(a) == len(b):  # same letters: keep the punctuation too (はい、 "here" vs はい "yes")
            return other if len((other or "").strip()) > len((preferred or "").strip()) else preferred
        return other if len(a) > len(b) else preferred
    return preferred if (preferred or "").strip() else other


def pick_sfx_read(tight: str, padded: str) -> str:
    """Hayai on the tight crop misreads hand-drawn sound effects (むに -> むり); the padded crop
    fixes most of those but sometimes stops early (ドクッ -> ド)."""
    return pick_read(tight, padded)


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
        crop = page.crop((max(0, x1), max(0, y1), min(W, x2), min(H, y2))).convert("RGB")
        tight, padded = (t.strip() for t in hayai_ocr.get(models_dir).read(
            [crop, _pad_mean(crop)], max_patches=512, max_new_tokens=64))
        return pick_sfx_read(tight, padded) or None
    pieces = []
    for line in lines:
        pad = max(2, int(0.1 * line.thickness))
        pieces.append(page.crop((max(0, line.x1 - pad), max(0, line.y1 - pad),
                                 min(W, line.x2 + pad), min(H, line.y2 + pad))))
    texts = [t.strip() for t in hayai_ocr.get(models_dir).read(pieces, max_patches=512, max_new_tokens=160)]
    texts = [t for t in texts if t]
    if not texts:
        return None
    if is_list(lines, texts):
        return LIST_SEP.join(texts)
    cjk = sum(_script_ratios(t)[1] + _script_ratios(t)[2] for t in texts) / len(texts) > 0.5
    return ("" if cjk else " ").join(texts)


LIST_SEP = " / "  # line breaks the translator keeps (chat comments, credits, status screens)


def is_list(lines: list, texts: list[str]) -> bool:
    """Horizontal lines that mostly start with a "name:" / "label:" prefix are a list (stream
    comments, credits, status readouts): one entry per line, never one flowing sentence."""
    if len(texts) < 2 or any(getattr(l, "vertical", False) for l in lines):
        return False
    labeled = sum(bool(re.match(r"^[^:：]{1,12}[:：]", t)) for t in texts)
    return labeled * 2 >= len(texts)


_SKIPPED_BBOXES: set[tuple] = set()  # blocks erased on purpose; not "left empty" errors


def install() -> None:
    import mangatl.core.pipeline as pipeline
    from mangatl.core.ml.model_manager import get_model_manager
    from mangatl.core.services import translation as tr
    from mangatl.utils.logging import log_message

    import mangatl.core.outside_text_processor as osb

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

        # Bubbles: Hayai matches manga-ocr on regular dialogue (OpenMantra ground truth: 0.9%
        # character error rate for both) but reads hand-drawn moans that manga-ocr turns into
        # unrelated words ("ぅ" -> "そして、").
        bubble_idx = [i for i, m in enumerate(bubble_metadata)
                      if i < len(texts) and not m.get("is_outside_text")]
        if bubble_idx and os.environ.get("MT_BUBBLE_OCR", "hayai") == "hayai":
            import hayai_ocr

            crops = tr._prepare_images_for_ocr([images_b64[i] for i in bubble_idx])
            ok = [(i, c) for i, c in zip(bubble_idx, crops) if c is not None]
            reads = hayai_ocr.get(models_dir).read([c.convert("RGB") for _, c in ok],
                                                   max_patches=512, max_new_tokens=160)
            for (i, _), t in zip(ok, reads):
                picked = pick_read(texts[i] if texts[i] != "[OCR FAILED]" else "", t.strip())
                if picked and picked != texts[i]:
                    texts[i] = picked
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
            log_message("Text re-read with Hayai OCR (outside bubbles: line by line via comic-text-detector):",
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

    import mangatl.core.outside_text_processor as osb

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
    import mangatl.core.image as image_pkg
    import mangatl.core.outside_text_processor as osb
    from mangatl.core.image import image_utils
    from mangatl.core.services import translation as tr

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
    # (thoughts) are marked by the parentheses already; italic markers around them only leave
    # a gap ("( I'm") in fonts without an italic
    text = text.replace(LIST_SEP.strip().join("  "), "\n")  # list lines (see is_list)
    text = re.sub(r"\(\s*\*+\s*", "(", text)
    text = re.sub(r"\s*\*+\s*\)", ")", text)
    t = re.sub(r'(["”’\'])\s+(?=["“‘\'])', r"\1\n", text)       # "…" "…"
    t = re.sub(r'([.!?~…♡])\s+(?=["“‘]|\'[A-Z])', r"\1\n", t)   # narration. "…" / 'Comment
    t = re.sub(r'(["”])\s+(?=[A-Z])', r"\1\n", t)               # "…" Narration
    t = re.sub(r'\)\s+(?=[(*"“‘A-Z])', ")\n", t)                # (thought) next
    t = re.sub(r'([.!?~…♡♥"”*])\s+(?=\()', r"\1\n", t)          # … (thought)
    return t


def install_narration_style() -> None:
    """Upstream sets every outside-bubble text like a sound effect: upper case, display font
    (Bangers), following the region's tilt. Long ones are narration/dialogue, which reads far
    better in the dialogue font, normal case, upright, one utterance per line."""
    import dataclasses

    import mangatl.core.pipeline as pipeline

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

    import mangatl.core as core
    import mangatl.core.pipeline as pipeline
    import mangatl.core.text as text_pkg
    import mangatl.core.text.placeholders as placeholders
    from mangatl.core.text import text_renderer
    from mangatl.utils.logging import log_message

    old = 'clean_text = " ".join(normalized_text.split())'
    new = 'clean_text = "\\n".join(" ".join(l.split()) for l in normalized_text.split("\\n")).strip()'
    src = textwrap.dedent(inspect.getsource(text_renderer.render_text_skia))
    preload = 'preload_hb_faces = {"regular": regular_hb_face}'
    if "_mt_symbol_hb" in text_renderer.__dict__ and src.count(preload) == 1:
        src = src.replace(preload, 'preload_hb_faces = {"regular": regular_hb_face, **_mt_symbol_hb()}')
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
    from mangatl.core.text import layout_engine as le

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
    import mangatl.core.pipeline as pipeline
    from mangatl.core.caching import get_cache
    from mangatl.utils.logging import log_message

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


# Closest stand-ins for characters lettering fonts often lack, tried in order
GLYPH_SUBSTITUTES = {
    "—": ["–", "--"], "–": ["-"], "―": ["—", "–", "--"], "~": ["〜", "-"], "〜": ["~", "-"],
    "‘": ["'"], "’": ["'"], "“": ['"'], "”": ['"'], "…": ["..."],
    "♥": ["♡"], "♡": ["♥"], "★": ["☆"], "☆": ["★"],
}


def install_glyph_substitutes() -> None:
    """Upstream drops every character the font has no glyph for (so "I—I" becomes "II" in fonts
    without an em dash). Swap such characters for a supported stand-in first; accented letters
    fall back to their plain form. Anything still unsupported is dropped as before."""
    import unicodedata

    from mangatl.core.text import font_manager, text_renderer

    original = font_manager.sanitize_text_for_font

    def substitute(text: str, font_path: str, verbose: bool = False) -> str:
        supported = font_manager.get_font_cmap(font_path)
        if text and supported:
            # A wave dash ("PLEASE~!", "Ahh~♡") has no good stand-in in the font: as "-" it
            # reads as a cut-off word and the line breaker splits there ("LIFE---" / "!!").
            # Draw it from the symbol font (as the tilde operator, which lettering fonts lack,
            # so it always takes that route), or drop it if there is no symbol font.
            if ord("~") not in supported:
                text = re.sub(r"[~〜]+", lambda m: "∼" * min(len(m.group()), 2) if _symbols_active else "",
                              text)
            out = []
            for ch in text:
                if ord(ch) in supported or ch.isspace():
                    out.append(ch)
                    continue
                options = GLYPH_SUBSTITUTES.get(ch, [])
                plain = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
                if plain and plain != ch:
                    options = options + [plain]
                out.append(next((o for o in options if all(ord(c) in supported for c in o)), ch))
            text = "".join(out)
        if _symbols_active and SYMBOL_CHARS & set(text):
            text = text.replace("♡", "♥")  # the outline heart is too thin next to lettering
            return "".join(seg if is_sym else original(seg, font_path, verbose)
                           for is_sym, seg in ((k, "".join(g)) for k, g in
                                               itertools.groupby(text, key=lambda c: c in SYMBOL_CHARS)))
        return original(text, font_path, verbose)

    font_manager.sanitize_text_for_font = substitute
    text_renderer.sanitize_text_for_font = substitute


def install_no_empty_bubbles() -> None:
    """Bubble rendering fixes. Upstream erases a bubble and leaves it blank when the translation
    does not fit at the minimum font size (tall, narrow bubbles, often after an OCR misread):
    retry at half the minimum size, and if it still does not fit, put the original lettering
    back. Colored / see-through bubbles also get outlined text."""
    import dataclasses

    import mangatl.core.pipeline as pipeline
    from mangatl.utils.logging import log_message

    original = pipeline._render_bubble
    original_config = pipeline._text_render_config
    outlined = threading.local()

    def text_render_config(config, sizes, is_outside_text):
        cfg = original_config(config, sizes, is_outside_text)
        if getattr(outlined, "on", False) and not is_outside_text:
            cfg = dataclasses.replace(cfg, outline_ratio=sizes.osb_outline_ratio)
        return cfg

    pipeline._text_render_config = text_render_config

    def render_bubble(image, bubble, text, bubble_id, render_info, page_cv, config, sizes,
                      processing_scale):
        # Colored / see-through bubbles (outline-only bubbles over the art, common in doujin CG)
        # are inpainted, not filled, so their text needs an outline like text on the art.
        outlined.on = bool(render_info and render_info.get("is_colored"))
        try:
            return _render(image, bubble, text, bubble_id, render_info, page_cv, config, sizes,
                           processing_scale)
        finally:
            outlined.on = False

    def _render(image, bubble, text, bubble_id, render_info, page_cv, config, sizes,
                processing_scale):
        rendered = original(image, bubble, text, bubble_id, render_info, page_cv, config,
                            sizes, processing_scale)
        if rendered is not None:
            return rendered
        smaller = sizes._replace(main_min_font=max(4, sizes.main_min_font // 2))
        rendered = original(image, bubble, text, bubble_id, render_info, page_cv, config,
                            smaller, processing_scale)
        if rendered is not None:
            log_message(f"Bubble {bubble_id}: fitted below the minimum font size",
                        always_print=True)
            return rendered
        if "original_crop_pil" in bubble:
            log_message(f"Bubble {bubble_id}: translation does not fit, kept the original "
                        f"lettering ('{text[:30]}')", always_print=True)
            return pipeline._paste_original_crop(image, bubble)
        return None

    pipeline._render_bubble = render_bubble


SYMBOL_FONT = Path(__file__).resolve().parent / "assets" / "symbols" / "DejaVuSans-Bold.ttf"
SYMBOL_CHARS = set("♡♥❤♪♫☆★♂♀→←↑↓⇒■□▶◀▲▼△▽●○◆◇※✓∼")
_symbols_active = False


def _split_symbols(text: str, style: str) -> list[tuple[str, str]]:
    runs = []
    for is_sym, chunk in itertools.groupby(text, key=lambda c: c in SYMBOL_CHARS):
        runs.append(("".join(chunk), "symbol" if is_sym else style))
    return runs


def install_symbol_fallback() -> None:
    """Hearts, notes and stars are everywhere in doujin text, but lettering fonts rarely have
    them, so upstream deletes them. Give those characters their own "symbol" style run, drawn
    and measured with DejaVu Sans Bold (bundled, free license); the dialogue font is used as
    is, never modified. Must run before install_keep_line_breaks (it re-executes the renderer)."""
    global _symbols_active
    import inspect
    import textwrap

    import mangatl.core.text as text_pkg
    from mangatl.core.text import drawing_engine, layout_engine, text_processing, text_renderer
    from mangatl.core.text.drawing_engine import load_font_resources
    from mangatl.utils.logging import log_message

    if not SYMBOL_FONT.exists():
        return

    def patch(module, func_name: str, old: str, new: str) -> bool:
        src = textwrap.dedent(inspect.getsource(getattr(module, func_name)))
        if src.count(old) != 1:
            log_message(f"Note: symbol-font patch for {func_name} no longer applies; skipped",
                        always_print=True)
            return False
        exec(compile(src.replace(old, new), module.__file__, "exec"), module.__dict__)
        return True

    styles = '("regular", "italic", "bold", "bold_italic")'
    resolve_end = "            return regular_typeface, regular_hb_face"
    ok = (patch(layout_engine, "calculate_styled_line_width", styles, styles[:-1] + ', "symbol")')
          and patch(layout_engine, "_styled_line_ink_path", styles, styles[:-1] + ', "symbol")')
          and patch(drawing_engine, "draw_layout", resolve_end,
                    '            if loaded_typefaces.get(style_name) is not None and '
                    'loaded_hb_faces.get(style_name) is not None:\n'
                    '                return loaded_typefaces[style_name], loaded_hb_faces[style_name]\n'
                    + resolve_end))
    if not ok:
        return
    text_renderer.draw_layout = text_pkg.draw_layout = drawing_engine.draw_layout

    original_parse = text_processing.parse_styled_segments

    def parse_styled_segments(text: str) -> list[tuple[str, str]]:
        out = []
        for seg, style in original_parse(text):
            out += _split_symbols(seg, style) if SYMBOL_CHARS & set(seg) else [(seg, style)]
        return out

    for module in (text_processing, text_renderer, drawing_engine, layout_engine, text_pkg):
        module.parse_styled_segments = parse_styled_segments

    def symbol_resources():
        _, typeface, hb_face = load_font_resources(str(SYMBOL_FONT), style="regular")
        return typeface, hb_face

    original_load = text_renderer._load_style_variants

    def load_style_variants(*args, **kwargs):
        typefaces, hb_faces = original_load(*args, **kwargs)
        typefaces["symbol"], hb_faces["symbol"] = symbol_resources()
        return typefaces, hb_faces

    text_renderer._load_style_variants = load_style_variants
    text_renderer._mt_symbol_hb = lambda: {"symbol": symbol_resources()[1]}
    _symbols_active = True


def install_keep_inner_quotes() -> None:
    """Upstream's response parser drops a leading and a trailing '"' from every item. That is
    right for an item the model wrapped in quotes, but narration items like
    '"Ah..." she said. "Hey."' lose their first and last quote. Put them back when the item
    contains further quotes (so they belong to the text)."""
    from mangatl.core.services import translation as tr

    original = tr._parse_llm_response_unified
    item = re.compile(r'^\s*(\d+)\s*[:.]\s*(.*?)\s*(?=\s*\n\s*\d+\s*[:.]|\n\s*\n|\s*\Z)',
                      re.MULTILINE | re.DOTALL)

    def parse(response_text, total_elements, provider, debug=False):
        texts = original(response_text, total_elements, provider, debug)
        raw = {int(n): t for n, t in item.findall(tr._strip_code_fences(response_text or ""))}
        for i, t in enumerate(texts):
            r = raw.get(i + 1, "")
            if r.startswith('"') and r.endswith('"') and r.count('"') > 2 and t == r[1:-1].strip():
                texts[i] = r
        return texts

    tr._parse_llm_response_unified = parse


def install_page_image() -> None:
    """With local OCR, upstream sends the translator text only. When MT_PAGE_IMAGE=1 (the model
    runs with its vision projector), attach the full page as well, so the translator sees who
    speaks, what is happening, and can catch OCR misreads (upstream's OCR-correction prompt
    already refers to the page image). Must run before anything wraps call_translation_api_batch."""
    import inspect
    import textwrap

    import mangatl.core.pipeline as pipeline
    from mangatl.core.services import translation as tr
    from mangatl.utils.logging import log_message

    if os.environ.get("MT_PAGE_IMAGE") != "1":
        return
    old = "page_attached = has_full_page and llm_ocr"
    src = textwrap.dedent(inspect.getsource(tr.call_translation_api_batch))
    if src.count(old) != 1:
        log_message("Note: page-image patch no longer applies; translating from text only",
                    always_print=True)
        return
    exec(compile(src.replace(old, "page_attached = has_full_page"), tr.__file__, "exec"), tr.__dict__)
    pipeline.call_translation_api_batch = tr.call_translation_api_batch


# Short items get a second look with the page image; see install_vision_short
VISION_SHORT_MAX = {"osb": 10, "bubble": 5}  # letters/kana/kanji in the source, punctuation not counted


def _words(text: str) -> str:
    """Text without markup, case, spacing or punctuation, to tell real changes from restyling."""
    return re.sub(r"[\W_]+", "", (text or "").lower())


def _short_item(ocr: str, osb: bool) -> bool:
    src = re.sub(r"^\[(OSB|Bubble)\]\s*", "", ocr or "").strip()
    if not src or src.upper().startswith("[OCR FAILED"):
        return False
    n = sum(c.isalnum() for c in src)
    return 1 <= n <= VISION_SHORT_MAX["osb" if osb else "bubble"]


def install_vision_short() -> None:
    """MT_VISION_SHORT=1: translate the page from text as usual, then show the model the page
    image and let it redo only the short items: sound effects, gasps, one- or two-word lines
    and cut-off words, whose meaning depends on the picture. Measured: the page image fixes
    those (sound effects, speaker of a short reply, OCR slips on two-kana effects) but adds
    meaning errors and pronoun guesses to longer dialogue and breaks the " / " separators of
    text blocks, so long items keep their text-only translation. Needs the vision projector.
    Install before install_text_dump so the dump records the final lines."""
    import inspect

    import mangatl.core.pipeline as pipeline
    from mangatl.core.services import translation as tr
    from mangatl.utils.logging import log_message

    if os.environ.get("MT_VISION_SHORT") != "1":
        return
    original = pipeline.call_translation_api_batch
    # the wrappers below this one take (*args, **kwargs); bind against upstream's own parameters
    signature = inspect.signature(tr.call_translation_api_batch)

    def call_translation_api_batch(*args, **kwargs):
        texts = original(*args, **kwargs)
        try:
            bound = signature.bind(*args, **kwargs)
            config = bound.arguments["config"]
            meta = bound.arguments.get("bubble_metadata") or []
            ocr = bound.arguments.get("ocr_texts_output") or []
            page_b64 = bound.arguments.get("full_image_b64")
            page_mime = bound.arguments.get("full_image_mime_type") or "image/jpeg"
            if not page_b64 or len(ocr) != len(texts) or len(meta) != len(texts):
                return texts
            short = [i for i, (o, m, t) in enumerate(zip(ocr, meta, texts))
                     if _short_item(o, bool(m.get("is_outside_text")))
                     and t and not t.strip().upper().startswith("[OCR FAILED")]
            if not short:
                return texts
            lines = []
            for i, (o, m, t) in enumerate(zip(ocr, meta, texts), 1):
                kind = "outside bubble" if m.get("is_outside_text") else "bubble"
                src = re.sub(r"^\[(OSB|Bubble)\]\s*", "", o or "").strip()
                lines.append(f"{i}: [{kind}] {src} || {t}")
            wanted = ", ".join(str(i + 1) for i in short)
            prompt = (
                "## CONTEXT\nAll text on this comic page in reading order, as "
                "`N: [where] source || draft translation`. The drafts were translated from the text "
                "alone. The image labeled `Full page` shows the page.\n\n## INPUT DATA\n"
                + "\n".join(lines)
                + f"\n\n## TASK\nItems {wanted} are short: sound effects, gasps, one-word lines, cut-off "
                "words. Their meaning depends on the picture. For each of them, find it on the page, "
                "see what is happening there (what makes the sound, who speaks, to whom), and give its "
                f"final {config.output_language} translation, keeping the draft when it is already right. "
                "The source of a short sound effect may be misread; go by the lettering on the page. "
                "Keep every other item's draft in mind so the lines fit together. Output only items "
                f"{wanted}, one per line as `N: translation`, nothing else."
                + tr._format_special_instructions(config)
            )
            system = tr._build_system_prompt_translation(
                config.output_language, mode="two-step", full_page_context=True,
                ero_doujinshi_mode=getattr(config, "ero_doujinshi_mode", False),
                input_language=config.input_language, llm_transcription=False)
            parts = [{"text": tr._FULL_PAGE_LABEL}, tr._image_part(page_mime, page_b64)]
            reply = tr._call_llm_endpoint(config, parts, prompt, system_prompt=system) or ""
        except Exception as e:  # never lose the page over the second look
            log_message(f"Short-item vision pass failed ({e}); keeping text-only lines",
                        always_print=True)
            return texts
        wanted_set = set(short)
        texts = list(texts)
        for line in tr._strip_code_fences(reply).splitlines():
            m = re.match(r"^\s*(\d+)\s*[:.]\s*(.+?)\s*$", line)
            if not m:
                continue
            i, new = int(m.group(1)) - 1, m.group(2)
            if "||" in new:  # echoed "source || translation"
                new = new.rsplit("||", 1)[1].strip()
            if (len(new) > 2 and new[0] == new[-1] == '"' and '"' not in new[1:-1]
                    and not texts[i].startswith('"')):  # quoted the line as dialogue
                new = new[1:-1]
            if i in wanted_set and new and _words(new) == _words(texts[i]):
                continue  # only markup, case or punctuation changed: keep the draft's styling
            if i in wanted_set and new and new != texts[i]:
                log_message(f"Vision, item {i + 1}: {texts[i]!r} -> {new!r}", always_print=True)
                texts[i] = new
        return texts

    pipeline.call_translation_api_batch = call_translation_api_batch


NOTES_MAX_LINES, NOTES_MAX_CHARS = 14, 1800
_REFUSAL = re.compile(r"\b(I can(?:no|')t|I'm sorry|I am sorry|I am unable|I'm unable|As an AI)\b", re.I)


def install_character_notes() -> None:
    """MT_CHAR_NOTES=<file>: running notes on the characters (name, stated gender, first-person
    pronoun, way of speaking, how they address others), sent with every page's translation
    request and updated after each page by a short text-only call. Keeps pronouns and voices
    consistent once the text has made them clear (a boy narrator stays "he" after the
    narration says so). The file persists, so the next chapter can start from it."""
    import dataclasses
    import inspect

    import mangatl.core.pipeline as pipeline
    from mangatl.core.services import translation as tr
    from mangatl.utils.logging import log_message

    path = os.environ.get("MT_CHAR_NOTES")
    if not path:
        return
    notes_file = Path(path)
    original = pipeline.call_translation_api_batch
    # the wrappers below this one take (*args, **kwargs); bind against upstream's own parameters
    signature = inspect.signature(tr.call_translation_api_batch)

    def read_notes() -> str:
        try:
            return notes_file.read_text().strip()
        except OSError:
            return ""

    def call_translation_api_batch(*args, **kwargs):
        notes = read_notes()
        bound = signature.bind(*args, **kwargs)
        config = bound.arguments["config"]
        if notes:
            extra = ("Character notes from earlier pages; follow them for names, pronouns and each "
                     "character's voice, unless this page clearly says otherwise:\n" + notes)
            noted = dataclasses.replace(
                config, special_instructions="\n".join(filter(None, [config.special_instructions, extra])))
            # swap the config where it was passed: the wrappers below read their other
            # arguments from kwargs, so the call must keep its shape
            if "config" in kwargs:
                kwargs["config"] = noted
            else:
                args = (noted, *args[1:])
        texts = original(*args, **kwargs)
        try:
            ocr = bound.arguments.get("ocr_texts_output") or []
            pairs = [f"{re.sub(r'^\[(OSB|Bubble)\]\s*', '', o or '').strip()} → {t}"
                     for o, t in zip(ocr, texts) if t and not str(t).startswith("[")]
            if not pairs:
                return texts
            adult = getattr(config, "ero_doujinshi_mode", False)
            system = ("You keep the character notes for a team translating a comic from "
                      f"{config.input_language} into {config.output_language}."
                      + (" The comic is 18+ adult fiction; record facts plainly, without comment."
                         if adult else ""))
            prompt = (
                f"## CURRENT NOTES\n{notes or '(none yet)'}\n\n## THIS PAGE (source → translation)\n"
                + "\n".join(pairs)
                + "\n\n## TASK\nUpdate the notes with what this page shows. One line per named "
                "character, using the translated name: gender only when the text states it (he/she "
                "in narration, 彼/彼女, 少年/少女, 兄/姉, what others call them), never from the name or "
                "from a guess; the first-person pronoun they use (俺, 僕, 私, わたくし, あたし...); how "
                "they speak (rough, polite, childish, archaic, mocking...); how they address the "
                f"others. Keep earlier facts unless this page contradicts them. At most "
                f"{NOTES_MAX_LINES} lines, each under 30 words. Output only the notes.")
            reply = tr._strip_code_fences(
                tr._call_llm_endpoint(config, [], prompt, system_prompt=system) or "").strip()
            reply = reply.replace("->", "→").replace("=>", "→")
            # drop placeholders like "(no named characters on this page)"
            lines = [ln for ln in reply.splitlines()
                     if ln.strip() and not re.fullmatch(r"\s*\(.*\)\s*", ln)]
            if (lines and len(lines) <= NOTES_MAX_LINES + 2 and len(reply) <= NOTES_MAX_CHARS
                    and not _REFUSAL.search(reply)):
                notes_file.write_text("\n".join(lines) + "\n")
                log_message(f"Character notes updated ({len(lines)} lines)", always_print=True)
            else:
                log_message("Character notes: reply unusable, keeping the previous notes",
                            always_print=True)
        except Exception as e:
            log_message(f"Character notes update failed ({e}); keeping the previous notes",
                        always_print=True)
        return texts

    pipeline.call_translation_api_batch = call_translation_api_batch


def install_text_dump() -> None:
    """Evaluation aid: with MT_DUMP_TEXT=<dir>, write <dir>/<page>.json listing every text
    region (box, outside-bubble flag, OCR read, translation) in reading order."""
    import json

    import mangatl.core.pipeline as pipeline

    out_dir = os.environ.get("MT_DUMP_TEXT")
    if not out_dir:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    _track_pages()
    original_batch = pipeline.call_translation_api_batch

    def call_translation_api_batch(*args, **kwargs):
        texts = original_batch(*args, **kwargs)
        meta = kwargs.get("bubble_metadata") or []
        ocr = kwargs.get("ocr_texts_output") or []
        items = [{"bbox": [int(v) for v in (m.get("original_bbox") or m.get("bbox") or [])],
                  "osb": bool(m.get("is_outside_text")),
                  "ocr": ocr[i] if i < len(ocr) else None,
                  "tl": texts[i] if i < len(texts) else None}
                 for i, m in enumerate(meta)]
        if _current_page():
            (Path(out_dir) / f"{_current_page()}.json").write_text(
                json.dumps(items, ensure_ascii=False, indent=1))
        return texts

    pipeline.call_translation_api_batch = call_translation_api_batch


_PAGE: dict = {"name": None, "tracking": False}


def _track_pages() -> None:
    """Remember which page is being processed (for text dumps and replay-miss logs)."""
    import mangatl.core.pipeline as pipeline

    if _PAGE["tracking"]:
        return
    _PAGE["tracking"] = True
    original = pipeline.translate_and_render

    def translate_and_render(image_path, *args, **kwargs):
        _PAGE["name"] = Path(image_path).stem
        _PAGE["label"] = f"{Path(image_path).parent.name}/{Path(image_path).name}"
        return original(image_path, *args, **kwargs)

    pipeline.translate_and_render = translate_and_render


def _current_page() -> str | None:
    return _PAGE["name"]


EMPHASIS_MIN_RATIO = 1.4  # source lettering this much above the page's typical size is emphasis
EMPHASIS_MAX_SCALE = float(os.environ.get("MT_EMPHASIS_MAX", 3.0))


def install_emphasis_sizes() -> None:
    """Upstream caps every bubble at one page-wide size taken from the typical source glyph,
    so a shout lettered three times bigger gets the same small English as a whisper, lost in
    its big bubble. Professional letterers follow the source: measure each bubble's own
    lettering and let bubbles with clearly bigger glyphs grow by the same ratio."""
    import types

    import cv2

    import mangatl.core.pipeline as pipeline
    from mangatl.core.image.image_utils import estimate_glyph_size
    from mangatl.utils.logging import log_message

    if os.environ.get("MT_EMPHASIS", "1") == "0":
        return
    state = types.SimpleNamespace(page=None, own={})
    original_tr = pipeline.translate_and_render

    def translate_and_render(*args, **kwargs):
        # long strips reuse the first page's size and skip _page_glyph_size: forget the
        # previous page's boxes, so they can't match bubbles on this one
        state.page, state.own = None, {}
        return original_tr(*args, **kwargs)

    pipeline.translate_and_render = translate_and_render
    original_page = pipeline._page_glyph_size

    def _page_glyph_size(page_image, bubbles_info, full_width):
        size = original_page(page_image, bubbles_info, full_width)
        gray = page_image if page_image.ndim == 2 else cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        own = {}
        for info in bubbles_info:
            if info.get("mask") is None or not info.get("bbox"):
                continue
            est = estimate_glyph_size(gray, info["bbox"], info["mask"], full_width)
            if est:
                own[tuple(int(v) for v in info["bbox"])] = est
        state.page, state.own = size, own
        return size

    pipeline._page_glyph_size = _page_glyph_size
    original_render = pipeline._render_bubble

    def _lookup(bbox):
        own = state.own
        key = tuple(int(v) for v in bbox)
        if key in own:
            return own[key]
        best, best_d = None, 1e9
        for k, v in own.items():
            d = sum(abs(a - b) for a, b in zip(k, key))
            if d < best_d:
                best, best_d = v, d
        return best if best_d <= 8 else None

    def _render_bubble(image, bubble, text, bubble_id, render_info, page_cv, config, sizes,
                       processing_scale):
        page = state.page
        est = _lookup(bubble.get("bbox", ())) if page else None
        # A lone "..." or "!!" is a few strokes; trust only measurements from enough ink
        if est and est[1] >= 3 and est[0] >= EMPHASIS_MIN_RATIO * page:
            # a letterer exaggerates a little: shouts read bigger than their glyph ratio
            scale = min((est[0] / page) ** 1.5, EMPHASIS_MAX_SCALE)
            bigger = round(sizes.main_max_font * scale)
            log_message(f"Bubble {bubble_id}: source lettering {est[0] / page:.1f}x the page's, "
                        f"max font {sizes.main_max_font} -> {bigger}", verbose=config.verbose)
            sizes = sizes._replace(main_max_font=bigger)
        return original_render(image, bubble, text, bubble_id, render_info, page_cv, config,
                               sizes, processing_scale)

    pipeline._render_bubble = _render_bubble


LAYOUT_KEEP_SHARE = 0.6  # a less clean layout must allow 1/0.6 = 1.67x bigger text to win


def install_layout_preferences() -> None:
    """Upstream's size search takes any trick that lets text grow: it hyphenates long words
    ("NEUROTOX-/IN") and moves trailing punctuation to its own line ("MOTHER" / "!!"). A
    letterer avoids both unless the text would otherwise get much smaller (a short "Fu..." in
    a narrow bubble). Lay out without the tricks first and only accept them when they allow
    clearly bigger text."""
    from mangatl.core.text import text_renderer

    original = text_renderer.find_optimal_layout
    MAX, HYPH, DETACH = 8, 10, 17  # positional indexes in find_optimal_layout
    names = {HYPH: "hyphenate_before_scaling", DETACH: "detach_trailing_punctuation"}

    def get(args, kwargs, i, default):
        return kwargs.get(names.get(i, ""), args[i] if len(args) > i else default)

    def run(args, kwargs, hyph, detach):
        a, k = list(args), dict(kwargs)
        for i, v in ((HYPH, hyph), (DETACH, detach)):
            if len(a) > i:
                a[i] = v
            else:
                k[names[i]] = v
        try:
            return original(*a, **k)
        except Exception:
            return None

    def find_optimal_layout(*args, **kwargs):
        may_hyph = get(args, kwargs, HYPH, True)
        may_detach = get(args, kwargs, DETACH, True)
        max_font = get(args, kwargs, MAX, 16)
        clean = run(args, kwargs, False, False)
        if clean is not None and clean.get("font_size", 0) >= max_font:
            return clean
        options = [clean]
        if may_hyph:
            options.append(run(args, kwargs, True, False))
        if may_detach:
            options.append(run(args, kwargs, False, True))
        if may_hyph and may_detach:
            options.append(run(args, kwargs, True, True))
        valid = [o for o in options if o is not None]
        if not valid:
            return original(*args, **kwargs)  # raises upstream's own error
        best = max(o.get("font_size", 0) for o in valid)
        for o in valid:  # in order of preference
            if o.get("font_size", 0) >= LAYOUT_KEEP_SHARE * best:
                return o
        return valid[0]

    text_renderer.find_optimal_layout = find_optimal_layout


def install_llm_cache() -> None:
    """Evaluation aid: with MT_LLM_CACHE=<dir>, LLM responses are stored by request content and
    replayed when the identical request comes again. llama.cpp isn't bit-exact between runs
    and one changed line changes the next pages' context, so without this, two runs of a
    typesetting change also differ in a third of their translations.

    With MT_LLM_FRESH=1, every request is sent with cache_prompt=false: llama.cpp then evaluates
    the whole prompt anew instead of reusing the previous request's prefix, which makes
    temperature-0 output repeatable (the reused prefix is what changes the floating-point
    results). Slower prompt processing; for prompt A/B runs.

    With MT_LLM_REPLAY=1 (regression harness, eval/harness.py), nothing is sent: a request that
    is not in the cache is appended to MT_LLM_MISSES (a .jsonl file) and fails at once with a
    non-retryable HTTP error. (Stopgap env switches; #7 replaces them with typed settings.)"""
    import hashlib
    import json
    import types

    from mangatl.utils.endpoints import _http

    cache_dir = os.environ.get("MT_LLM_CACHE")
    fresh = os.environ.get("MT_LLM_FRESH") == "1"
    replay = os.environ.get("MT_LLM_REPLAY") == "1"
    misses = os.environ.get("MT_LLM_MISSES")
    if replay and not cache_dir:
        raise RuntimeError("MT_LLM_REPLAY=1 needs MT_LLM_CACHE")
    if replay:
        _track_pages()
    if not (cache_dir or fresh):
        return
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    real = _http.requests

    class _Cached:
        status_code = 200

        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

    def post(url, headers=None, json=None, timeout=None, **kwargs):  # noqa: A002
        if fresh and isinstance(json, dict):
            json = {**json, "cache_prompt": False}
        if not cache_dir:
            return real.post(url, headers=headers, json=json, timeout=timeout, **kwargs)
        key = hashlib.sha256(_json.dumps([url, json], sort_keys=True).encode()).hexdigest()
        f = Path(cache_dir) / f"{key}.json"
        if f.exists():
            return _Cached(_json.loads(f.read_text()))
        if replay:
            if misses:
                with open(misses, "a", encoding="utf-8") as fh:
                    fh.write(_json.dumps({"key": key, "page": _PAGE.get("label"), "request": _strip_images(json)},
                                         ensure_ascii=False) + "\n")
            miss = types.SimpleNamespace(status_code=418, text=f"replay miss {key[:12]}")
            raise real.exceptions.HTTPError(f"replay miss {key[:12]}", response=miss)
        response = real.post(url, headers=headers, json=json, timeout=timeout, **kwargs)
        response.raise_for_status()
        f.write_text(_json.dumps(response.json(), ensure_ascii=False))
        # the request too, minus image data, to see exactly what the model was asked
        f.with_suffix(".req.json").write_text(
            _json.dumps(_strip_images(json), ensure_ascii=False, indent=1))
        return response

    def _strip_images(obj):
        if isinstance(obj, dict):
            return {k: _strip_images(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_strip_images(v) for v in obj]
        if isinstance(obj, str) and obj.startswith("data:image"):
            return obj[:40] + "...(image)"
        return obj

    _json = json
    _http.requests = types.SimpleNamespace(post=post, exceptions=real.exceptions)


def _junk_read(ocr: str) -> bool:
    """An outside-text "read" with no Japanese and at most 3 letters is speed lines or
    artwork the detector mistook for lettering ("H")."""
    t = re.sub(r"^\[(OSB|Bubble)\]\s*", "", ocr or "").strip()
    if not t or t.startswith("["):
        return False
    _, kana, han = _script_ratios(t)
    return kana + han == 0 and sum(c.isalnum() for c in t) <= 3 and not set(t) <= set("!?.…")


def install_keep_junk_osb() -> None:
    """Leave outside-text regions whose OCR is junk untouched (original art pasted back)
    instead of drawing the junk on the page."""
    import mangatl.core.pipeline as pipeline

    original = pipeline._render_action

    def _render_action(bubble, text, provider_tag):
        action = original(bubble, text, provider_tag)
        if (action == "render" and bubble.get("is_outside_text") and "original_crop_pil" in bubble
                and _junk_read(bubble.get("ocr_text", ""))):
            return "keep_original"
        return action

    pipeline._render_action = _render_action


# Sound effects: source all kana (plus small-tsu, long marks, hearts, punctuation)
_KANA_SFX = re.compile(r"^[ぁ-ゖァ-ヺーヽヾ・ッっー〜~♡♥!！?？.…、。\s]+$")
SFX_LABEL_SCALE = (1.1, 2.2)  # label size range, relative to the page's dialogue font size
SFX_LABEL_SHARE = 0.28  # ...within which it follows the effect's size (share of its longer side)


def _sfx_label(ocr: str, translation: str) -> str | None:
    """The label for an outside-bubble sound effect, or None for anything else. The translator
    styles sound effects as **BOLD** and mimetic effects as *italic* (upstream's prompt); a
    label needs that styling, a kana-only source, and one English word (possibly repeated:
    "twitch twitch"). Spoken short lines ("*No way...*", "*Guh...*") don't qualify."""
    src = re.sub(r"^\[(OSB|Bubble)\]\s*", "", ocr or "").strip()
    if not src or not _KANA_SFX.match(src) or not re.search(r"[ぁ-ヺ]", src):
        return None
    m = re.fullmatch(r"(\*{1,3})([^*]+)\1[♡♥!]*", (translation or "").strip())
    if not m:
        return None
    inner = m.group(2).strip()
    if re.search(r"\.\.\.|…|[?!]", inner) or re.sub(r"[A-Za-z'\-\s,]", "", inner):
        return None
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", inner)
    if not words or len({w.lower() for w in words}) != 1:
        return None
    return inner.upper()


def _place_label(size: tuple[int, int], box, page_wh, taken, gray) -> tuple[int, int]:
    """Top-left corner for a label of size (w, h) next to box: the side with no other text
    and the calmest background, preferring right or left of tall effects and below or above
    wide ones. Falls back to the centre of the effect."""
    import numpy as np

    w, h = size
    x0, y0, x1, y1 = box
    W, H = page_wh
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    gap = max(4, h // 4)
    sides = {"right": (x1 + gap, cy - h // 2), "left": (x0 - gap - w, cy - h // 2),
             "below": (cx - w // 2, y1 + gap), "above": (cx - w // 2, y0 - gap - h)}
    order = (["right", "left", "below", "above"] if (y1 - y0) > (x1 - x0)
             else ["below", "above", "right", "left"])

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    def crosses_border(side, x, y) -> bool:
        """A panel border between the effect and the label: a row (or column) that is dark
        nearly all the way across the label, in the strip from the effect's edge to the label's
        far side. Gutters are plain white and would otherwise look like the calmest spot."""
        if gray is None:
            return False
        if side in ("below", "above"):
            strip = gray[y1:y + h, x:x + w] if side == "below" else gray[y:y0, x:x + w]
            lines = (strip < 80).mean(axis=1) if strip.size else []
        else:
            strip = gray[y:y + h, x1:x + w] if side == "right" else gray[y:y + h, x:x0]
            lines = (strip < 80).mean(axis=0) if strip.size else []
        return bool(len(lines)) and float(np.max(lines)) > 0.9

    best, best_score = None, None
    for rank, side in enumerate(order):
        x, y = sides[side]
        if x < 0 or y < 0 or x + w > W or y + h > H:
            continue
        r = (x, y, x + w, y + h)
        if any(overlap(r, t) for t in taken) or crosses_border(side, x, y):
            continue
        busy = float(np.std(gray[y:y + h, x:x + w])) if gray is not None else 0.0
        score = busy + 6 * rank
        if best_score is None or score < best_score:
            best, best_score = (x, y), score
    if best is None:
        best = (min(max(0, cx - w // 2), W - w), min(max(0, cy - h // 2), H - h))
    return best


def install_sfx_labels() -> None:
    """MT_SFX=label: like a professional scanlation, keep the artist's sound-effect lettering
    and add a small outlined English label beside it, instead of erasing it and drawing the
    English over the cleaned art. Only outside-bubble sound effects (see _sfx_label); speech,
    moans and narration are drawn as usual."""
    import numpy as np
    from PIL import ImageDraw, ImageFont

    import mangatl.core.pipeline as pipeline
    from mangatl.core.text.font_manager import find_font_variants
    from mangatl.utils.logging import log_message

    if os.environ.get("MT_SFX") != "label":
        return
    original_action = pipeline._render_action
    original_render = pipeline._render_translations

    def _render_action(bubble, text, provider_tag):
        action = original_action(bubble, text, provider_tag)
        if bubble.get("_sfx_label") and action not in ("ocr_failed", "invalid"):
            return "keep_original"
        return action

    def _render_translations(image, sorted_bubble_data, translated_texts, processed_bubbles_info,
                             page_cv, config, sizes, *args, **kwargs):
        labels = []
        for bubble, text in zip(sorted_bubble_data, translated_texts):
            if bubble.get("is_outside_text") and "original_crop_pil" in bubble:
                label = _sfx_label(bubble.get("ocr_text", ""), text)
                if label:
                    bubble["_sfx_label"] = label
                    labels.append((bubble["bbox"], label))
        image = original_render(image, sorted_bubble_data, translated_texts, processed_bubbles_info,
                                page_cv, config, sizes, *args, **kwargs)
        if not labels:
            return image
        font_dir = config.outside_text.osb_font_dir or config.rendering.font_dir
        variants = find_font_variants(font_dir)
        font_path = variants.get("bold") or variants.get("regular")
        base = max(sizes.main_min_font, sizes.main_max_font)
        mode = image.mode
        page = image.convert("RGB")
        gray = np.asarray(page.convert("L"), dtype=np.float32)
        taken = [tuple(int(v) for v in b["bbox"]) for b in sorted_bubble_data
                 if not b.get("_sfx_label")]
        draw = ImageDraw.Draw(page)
        for bbox, label in labels:
            box = tuple(int(v) for v in bbox)
            # big effects get big labels, like a letterer's
            side = max(box[2] - box[0], box[3] - box[1])
            size = round(min(max(side * SFX_LABEL_SHARE, base * SFX_LABEL_SCALE[0]),
                             base * SFX_LABEL_SCALE[1]))
            font = ImageFont.truetype(str(font_path), size)
            stroke = max(2, round(size * 0.14))
            l, t, r, b = draw.textbbox((0, 0), label, font=font, stroke_width=stroke)
            x, y = _place_label((r - l, b - t), box, page.size, taken, gray)
            draw.text((x - l, y - t), label, font=font, fill=(0, 0, 0),
                      stroke_width=stroke, stroke_fill=(255, 255, 255))
            taken.append((x, y, x + r - l, y + b - t))
            log_message(f"SFX label {label!r} beside {box}", verbose=config.verbose)
        return page.convert(mode) if mode != "RGB" else page

    pipeline._render_action = _render_action
    pipeline._render_translations = _render_translations


LINE_HEIGHT_EM = 1.0  # minimum baseline-to-baseline distance, in font sizes


def install_min_line_height() -> None:
    """Lettering fonts disagree on built-in line spacing: CC Wild Words is 0.86 em (lines of
    capitals nearly touch), Comic Neue and Anime Ace are 1.15-1.25. Open tight fonts up to
    LINE_HEIGHT_EM so every font gets the airy leading of hand lettering."""
    import skia

    from mangatl.core.text import text_renderer

    original = text_renderer.find_optimal_layout
    TYPEFACE, SPACING, VERTICAL = 4, 9, 18  # positional indexes in find_optimal_layout
    cache: dict = {}

    def line_em(typeface) -> float:
        key = id(typeface)
        if key not in cache:
            m = skia.Font(typeface, 100).getMetrics()
            cache[key] = (-m.fAscent + m.fDescent + m.fLeading) / 100 or 1.0
        return cache[key]

    def find_optimal_layout(*args, **kwargs):
        args = list(args)
        vertical = kwargs.get("vertical_stack", args[VERTICAL] if len(args) > VERTICAL else False)
        typeface = kwargs.get("regular_typeface", args[TYPEFACE] if len(args) > TYPEFACE else None)
        if not vertical and typeface is not None:
            factor = max(1.0, LINE_HEIGHT_EM / line_em(typeface))
            if len(args) > SPACING:
                args[SPACING] *= factor
            else:
                kwargs["line_spacing_mult"] = kwargs.get("line_spacing_mult", 1.0) * factor
        return original(*args, **kwargs)

    text_renderer.find_optimal_layout = find_optimal_layout


VERTICAL_MIN_GAIN = 1.8  # stack letters vertically only if that allows text this much bigger


def install_vertical_only_if_much_bigger() -> None:
    """Upstream stacks short Latin text letter by letter ("W/H/A/T") in tall regions whenever
    that is at least as big as a horizontal line. Letterers set English horizontally and only
    stack when a horizontal line would be tiny (a long moan along a narrow strip)."""
    from mangatl.core.text import text_renderer

    original = text_renderer._auto_vertical_layout_is_better

    from mangatl.utils.logging import log_message

    def better(horizontal_layout, vertical_layout, max_render_height):
        if not original(horizontal_layout, vertical_layout, max_render_height):
            return False
        h, v = horizontal_layout.get("font_size", 0), vertical_layout.get("font_size", 0)
        log_message(f"Vertical stacking candidate: size {v} vs horizontal {h}", verbose=False)
        return v >= VERTICAL_MIN_GAIN * h

    text_renderer._auto_vertical_layout_is_better = better
