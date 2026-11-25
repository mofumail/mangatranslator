"""Second look at short items with the page image (--vision short)."""

from __future__ import annotations

import re

from mangatl.translate import service as tr
from mangatl.common.log import log_message

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


def revise_short_items(config, bubble_metadata: list[dict], ocr_texts: list[str], texts: list[str],
                       full_image_b64: str | None, full_image_mime_type: str | None) -> list[str]:
    """translation.vision = "short": translate the page from text as usual, then show the model the page
    image and let it redo only the short items: sound effects, gasps, one- or two-word lines
    and cut-off words, whose meaning depends on the picture. Measured: the page image fixes
    those (sound effects, speaker of a short reply, OCR slips on two-kana effects) but adds
    meaning errors and pronoun guesses to longer dialogue and breaks the " / " separators of
    text blocks, so long items keep their text-only translation. Needs the vision projector."""
    if config.translation.vision != "short":
        return texts
    try:
        meta = bubble_metadata or []
        ocr = ocr_texts or []
        page_b64 = full_image_b64
        page_mime = full_image_mime_type or "image/jpeg"
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
            f"final {config.translation.target_language} translation, keeping the draft when it is already right. "
            "The source of a short sound effect may be misread; go by the lettering on the page. "
            "Keep every other item's draft in mind so the lines fit together. Output only items "
            f"{wanted}, one per line as `N: translation`, nothing else."
            + tr._format_special_instructions(config)
        )
        system = tr._build_system_prompt_translation(
            config.translation.target_language, full_page_context=True,
            ero_doujinshi_mode=config.translation.adult,
            input_language=config.translation.source_language)
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

