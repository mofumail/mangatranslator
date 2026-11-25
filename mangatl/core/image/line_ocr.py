"""Japanese OCR helpers: line-by-line reading of text outside bubbles, read picking, scripts.

manga-ocr squashes its input to 224x224, so a narration block with many lines turns into a few
pixels per glyph and it "reads" fluent but invented text. Instead, the established design from
manga-image-translator and BallonsTranslator is used: detect the individual text lines
(comic-text-detector, mangatl.core.ml.text_lines), read each line with a recognizer that keeps the
aspect ratio (Hayai OCR, mangatl.core.ml.hayai_ocr), and join them in reading order. Regions with
fewer lines (sound effects, short asides) are read whole by Hayai, which was trained on
onomatopoeia. Used by services.translation._perform_manga_ocr.
"""

from __future__ import annotations

import itertools
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
# Pages are registered when the outside-text regions are built (same coordinates).
_page_lock = threading.Lock()
_page_ids = itertools.count(1)
_PAGES: "OrderedDict[int, object]" = OrderedDict()
_PAGE_LINES: dict[int, list] = {}


def register_page(pil_image) -> int:
    with _page_lock:
        token = next(_page_ids)
        _PAGES[token] = pil_image
        while len(_PAGES) > 4:  # a few pages in flight at most
            old, _ = _PAGES.popitem(last=False)
            _PAGE_LINES.pop(old, None)
        return token


def _page_lines(token: int, models_dir: Path) -> list:
    from mangatl.core.ml import text_lines

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
    from mangatl.core.ml import hayai_ocr, text_lines

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
