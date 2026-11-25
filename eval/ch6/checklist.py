"""Known-hard lines of the reference chapter, side by side for several runs.

  checklist.py <label> [label ...]

Each entry is (page, a piece of the Japanese OCR, what a good translation does). The lines are
found by their OCR text, so runs with different region order still line up. Judge by eye; the
"good" column says what to look for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CHECKS = [
    ("001", "ルダク", "name: Rudak (pro)"),
    ("001", "第1話", "title name: Priscilla"),
    ("002", "まっ", "cut-off start of a line, not a sigh"),
    ("002", "おと少しで", "split sentence, part 1 ('I was this close')"),
    ("002", "ディアンヌ", "split sentence, part 2 ('to killing Dianne')"),
    ("003", "悪ィな", "rough voice ('Sorry about this')"),
    ("003", "オレのために", "natural ('Die for me')"),
    ("003", "リーフさんが", "honorific handling"),
    ("004", "邪魔者", "cut off by the next speaker"),
    ("004", "させない", "interruption: her reply 'I won't let you', not the end of his line"),
    ("004", "お母様は絶対", "split: 'I'll never let you...'"),
    ("004", "殺させない", "split: '...kill Mother!!'"),
    ("004", "家族愛", "sarcastic ('How heartwarming')"),
    ("004", "大いに結構", "sarcastic ('Family love is wonderful')"),
    ("005", "死ねェ", "stretched shout, part 1 ('Die!')"),
    ("005", "エ", "stretched shout continues ('Die...'), not 'E-'"),
    ("008", "アカ", "SFX: blush / flush"),
    ("008", "アヤ", "SFX misread (アセ?): fluster"),
    ("011", "人間が持つ", "split sentence, part 1"),
    ("011", "活性化", "split sentence, part 2"),
    ("016", "プリシラ", "name: Priscilla"),
    ("017", "アセ", "SFX: sweat / fluster"),
]


def load(label: str) -> dict[str, list]:
    d = HERE / "out" / label / "text"
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}


def main() -> None:
    labels = sys.argv[1:]
    runs = {lb: load(lb) for lb in labels}
    width = max(len(lb) for lb in labels)
    for page, jp, good in CHECKS:
        print(f"\n{page} {jp}  -- {good}")
        for lb in labels:
            items = runs[lb].get(page, [])
            hits = [it for it in items if jp in (it["ocr"] or "")]
            if jp == "エ":  # the stretched-vowel bubbles: every item that is only エ and !
                hits = [
                    it
                    for it in items
                    if (it["ocr"] or "").strip("!！ ") and set((it["ocr"] or "").strip()) <= set("エェ!！ ")
                ]
            text = " / ".join(repr(it["tl"]) for it in hits) or "(not found)"
            print(f"  {lb:<{width}} {text}")


if __name__ == "__main__":
    main()
