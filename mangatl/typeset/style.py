"""How one region's text is lettered, beyond the automatic fit: font, size, case, emphasis,
alignment, spacing, rotation, colour, outline and placement.

A region's style (Region.style in the page document) names a preset of the chapter's style sheet
(typesetting.presets) and overrides any of its fields; unset fields fall through to the preset,
then to the automatic lettering. A region without a style renders exactly as before.

    LetterStyle(preset="shout")                       # the sheet's shout lettering
    LetterStyle(preset="dialogue", scale=1.2, align="left")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HEX = r"^#[0-9a-fA-F]{6}$"


class LetterStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: str | None = None  # a preset of the chapter's style sheet
    font: str | None = None  # font pack in fonts/
    size: float | None = Field(default=None, gt=0)  # the largest size (px) the text is fitted at
    scale: float | None = Field(default=None, gt=0)  # ...or that size relative to the page's dialogue size
    caps: bool | None = None  # set in capitals
    bold: bool | None = None  # the whole text bold (spans: **...** in the text)
    italic: bool | None = None  # the whole text italic (spans: *...*)
    align: Literal["center", "left", "right"] | None = None
    line_spacing: float | None = Field(default=None, gt=0)  # multiple of the font's line height
    rotation: float | None = Field(default=None, ge=-90, le=90)  # degrees, clockwise
    color: str | None = Field(default=None, pattern=HEX)  # text colour, "#rrggbb"
    outline: float | None = Field(default=None, ge=0)  # outline width in px (0: none)
    outline_color: str | None = Field(default=None, pattern=HEX)
    box: tuple[int, int, int, int] | None = (
        None  # the text area on the page (x1, y1, x2, y2), not the bubble's
    )
    offset: tuple[float, float] | None = None  # move the text block by (dx, dy) px
    padding: float | None = Field(default=None, ge=0)  # margin to the bubble's edge (px)

    def over(self, base: LetterStyle | None) -> LetterStyle:
        """This style on top of `base`: fields set here win."""
        if base is None:
            return self
        return base.model_copy(update=self.model_dump(exclude_none=True))

    def is_empty(self) -> bool:
        return not self.model_dump(exclude_none=True)


def resolve(
    region_style: dict | LetterStyle | None, presets: dict[str, LetterStyle], kind: str
) -> LetterStyle | None:
    """The style a region renders with: its own fields over its preset (default: "dialogue" for
    bubbles, "outside" for text on the art, when the sheet has one). None: nothing to apply."""
    own = LetterStyle.model_validate(region_style) if isinstance(region_style, dict) else region_style
    name = own.preset if own is not None and own.preset else ("outside" if kind == "outside" else "dialogue")
    preset = presets.get(name)
    if own is None and preset is None:
        return None
    style = (own or LetterStyle()).over(preset)
    return None if style.model_copy(update={"preset": None}).is_empty() else style


def styled_text(text: str, style: LetterStyle | None, em_dash: bool) -> str:
    """The text as drawn: "--" as an em dash (em_dash), capitals, whole-text emphasis (unless the
    text marks its own spans)."""
    if em_dash:
        text = re.sub(r"(?<!-)--(?!-)", "—", text)
    if style is None:
        return text
    if style.caps:
        text = text.upper()
    if "*" not in text:
        marker = "***" if style.bold and style.italic else "**" if style.bold else "*" if style.italic else ""
        if marker:
            text = "\n".join(f"{marker}{line}{marker}" if line.strip() else line for line in text.split("\n"))
    return text


def rgb(color: str | None) -> tuple[int, int, int] | None:
    if color is None:
        return None
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def font_packs(fonts_dir: Path) -> dict[str, list[str]]:
    """Font packs (folders of .ttf/.otf) and the styles each has."""
    from mangatl.typeset.fonts import find_font_variants

    packs = {}
    for d in sorted(p for p in fonts_dir.iterdir() if p.is_dir()):
        try:
            variants = find_font_variants(str(d), verbose=False)
        except Exception:
            continue
        packs[d.name] = sorted(k for k, v in variants.items() if v)
    return packs
