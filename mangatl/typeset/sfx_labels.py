"""Sound-effect labels (typesetting.sfx = "label", the default).

Like a professional scanlation, keep the artist's sound-effect lettering and add a small outlined
English label beside it, instead of erasing it and drawing the English over the cleaned art.
Only outside-bubble sound effects (see _sfx_label); speech, moans and narration are drawn as usual.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import ImageDraw, ImageFont

from mangatl.common.log import log_message
from mangatl.typeset.fonts import find_font_variants

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
    sides = {
        "right": (x1 + gap, cy - h // 2),
        "left": (x0 - gap - w, cy - h // 2),
        "below": (cx - w // 2, y1 + gap),
        "above": (cx - w // 2, y0 - gap - h),
    }
    order = (
        ["right", "left", "below", "above"] if (y1 - y0) > (x1 - x0) else ["below", "above", "right", "left"]
    )

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    def crosses_border(side, x, y) -> bool:
        """A panel border between the effect and the label: a row (or column) that is dark
        nearly all the way across the label, in the strip from the effect's edge to the label's
        far side. Gutters are plain white and would otherwise look like the calmest spot."""
        if gray is None:
            return False
        if side in ("below", "above"):
            strip = gray[y1 : y + h, x : x + w] if side == "below" else gray[y:y0, x : x + w]
            lines = (strip < 80).mean(axis=1) if strip.size else []
        else:
            strip = gray[y : y + h, x1 : x + w] if side == "right" else gray[y : y + h, x:x0]
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
        busy = float(np.std(gray[y : y + h, x : x + w])) if gray is not None else 0.0
        score = busy + 6 * rank
        if best_score is None or score < best_score:
            best, best_score = (x, y), score
    if best is None:
        best = (min(max(0, cx - w // 2), W - w), min(max(0, cy - h // 2), H - h))
    return best


def mark_labels(sorted_bubble_data: list[dict], translated_texts: list[str]) -> list[tuple]:
    """Mark sound effects that get a label (`_sfx_label`: their original lettering is kept) and
    return (bbox, label) for each."""
    labels = []
    for bubble, text in zip(sorted_bubble_data, translated_texts):
        if bubble.get("is_outside_text") and "original_crop_pil" in bubble:
            label = _sfx_label(bubble.get("ocr_text", ""), text)
            if label:
                bubble["_sfx_label"] = label
                labels.append((bubble["bbox"], label))
    return labels


def draw_labels(image, labels: list[tuple], sorted_bubble_data: list[dict], config, sizes):
    """Draw the labels beside their effects, sized to the effect, away from other text."""
    if not labels:
        return image
    font_dir = config.typesetting.outside.font_dir or config.typesetting.bubbles.font_dir
    variants = find_font_variants(font_dir)
    font_path = variants.get("bold") or variants.get("regular")
    base = max(sizes.main_min_font, sizes.main_max_font)
    mode = image.mode
    page = image.convert("RGB")
    gray = np.asarray(page.convert("L"), dtype=np.float32)
    taken = [tuple(int(v) for v in b["bbox"]) for b in sorted_bubble_data if not b.get("_sfx_label")]
    draw = ImageDraw.Draw(page)
    for bbox, label in labels:
        box = tuple(int(v) for v in bbox)
        # big effects get big labels, like a letterer's
        side = max(box[2] - box[0], box[3] - box[1])
        size = round(min(max(side * SFX_LABEL_SHARE, base * SFX_LABEL_SCALE[0]), base * SFX_LABEL_SCALE[1]))
        font = ImageFont.truetype(str(font_path), size)
        stroke = max(2, round(size * 0.14))
        l, t, r, b = draw.textbbox((0, 0), label, font=font, stroke_width=stroke)
        x, y = _place_label((r - l, b - t), box, page.size, taken, gray)
        draw.text(
            (x - l, y - t), label, font=font, fill=(0, 0, 0), stroke_width=stroke, stroke_fill=(255, 255, 255)
        )
        taken.append((x, y, x + r - l, y + b - t))
        log_message(f"SFX label {label!r} beside {box}", verbose=config.verbose)
    return page.convert(mode) if mode != "RGB" else page
