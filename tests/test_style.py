"""Region styles: presets, overrides, and the text as drawn."""

import pytest

from mangatl.typeset.style import LetterStyle, resolve, rgb, styled_text

PRESETS = {
    "dialogue": LetterStyle(font="CC Wild Words", caps=True),
    "shout": LetterStyle(font="CC Wild Words", caps=True, bold=True, scale=1.3),
}


def test_a_region_without_style_takes_the_kind_default():
    assert resolve(None, PRESETS, "bubble") == PRESETS["dialogue"]
    assert resolve(None, PRESETS, "outside") is None  # no "outside" preset: automatic lettering
    assert resolve(None, {}, "bubble") is None  # no sheet, no style: exactly as before


def test_own_fields_win_over_the_preset():
    s = resolve({"preset": "shout", "scale": 1.5, "align": "left"}, PRESETS, "bubble")
    assert s is not None and (s.font, s.bold, s.scale, s.align) == ("CC Wild Words", True, 1.5, "left")


def test_styled_text():
    assert styled_text("Wait--", None, em_dash=True) == "Wait—"
    assert styled_text("a---b", None, em_dash=True) == "a---b"  # not a pair of dashes
    assert styled_text("No way!", LetterStyle(caps=True, bold=True), em_dash=False) == "**NO WAY!**"
    assert styled_text("Two\nlines", LetterStyle(italic=True), em_dash=False) == "*Two*\n*lines*"
    assert styled_text("a **word**", LetterStyle(italic=True), em_dash=False) == "a **word**"  # own spans


def test_validation():
    assert rgb("#ff8000") == (255, 128, 0)
    with pytest.raises(ValueError):
        LetterStyle.model_validate({"color": "red"})
    with pytest.raises(ValueError):
        LetterStyle.model_validate({"sizes": 20})  # unknown field
