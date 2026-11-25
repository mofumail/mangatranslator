"""Outside-bubble text that is narration or dialogue printed on the art, not a sound effect.

Such text reads far better in the dialogue font, normal case, upright, one utterance per line,
than set like a sound effect (upper case, display font, following the region's tilt).
Regions are marked by mark_narration after translation; the renderer
(pipeline._render_osb_attempt / _osb_render_attempts) honours the marks.
"""

from __future__ import annotations

import re

from mangatl.core.image.line_ocr import LIST_SEP

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


def mark_narration(texts: list[str], bubble_metadata: list[dict]) -> None:
    """Mark long outside-bubble translations as narration (`mt_narration`, `mt_text`)."""
    for i, m in enumerate(bubble_metadata):
        if i < len(texts) and m.get("is_outside_text") and isinstance(texts[i], str):
            t = texts[i].strip()
            if len(t.split()) >= NARRATION_MIN_WORDS and not t.startswith("["):
                m["mt_narration"] = True
                m["mt_text"] = utterance_lines(t)
