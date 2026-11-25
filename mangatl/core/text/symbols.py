"""Hearts, notes and stars drawn from a bundled symbol font.

They are everywhere in doujin text, but lettering fonts rarely have them, so they used to be
deleted. Those characters get their own "symbol" style run, drawn and measured with DejaVu Sans
Bold (bundled, free license, assets/symbols/); the dialogue font is used as is, never modified.
"""

from __future__ import annotations

import itertools
from mangatl.settings import ROOT

SYMBOL_FONT = ROOT / "assets" / "symbols" / "DejaVuSans-Bold.ttf"
SYMBOL_CHARS = set("♡♥❤♪♫☆★♂♀→←↑↓⇒■□▶◀▲▼△▽●○◆◇※✓∼")


def active() -> bool:
    return SYMBOL_FONT.exists()


def split_symbols(text: str, style: str) -> list[tuple[str, str]]:
    runs = []
    for is_sym, chunk in itertools.groupby(text, key=lambda c: c in SYMBOL_CHARS):
        runs.append(("".join(chunk), "symbol" if is_sym else style))
    return runs


def symbol_resources():
    """(typeface, HarfBuzz face) of the symbol font."""
    from mangatl.core.text.drawing_engine import load_font_resources

    _, typeface, hb_face = load_font_resources(str(SYMBOL_FONT), style="regular")
    return typeface, hb_face
