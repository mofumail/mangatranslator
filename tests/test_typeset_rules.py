"""Typesetting rules that need no fonts or models."""

from mangatl.typeset.narration import NARRATION_MIN_WORDS, mark_narration, utterance_lines
from mangatl.typeset.sfx_labels import _sfx_label
from mangatl.typeset.symbols import split_symbols


def test_sfx_label_needs_kana_source_styling_and_one_word():
    assert _sfx_label("[OSB] ドドドッ", "**BAM**") == "BAM"
    assert _sfx_label("ピクピク", "*twitch twitch*") == "TWITCH TWITCH"
    assert _sfx_label("ズキン♡", "*throb*♡") == "THROB"


def test_sfx_label_rejects_speech_and_kanji():
    assert _sfx_label("うそ…", "*No way...*") is None  # spoken, with an ellipsis
    assert _sfx_label("ドン", "BAM") is None  # not styled as an effect
    assert _sfx_label("轟", "**RUMBLE**") is None  # kanji source
    assert _sfx_label("ガタッ", "**CLATTER BANG**") is None  # two different words


def test_utterance_lines_one_quote_per_line():
    t = '"Ah..." she said. "Hey." (Me?) Then it rained.'
    assert utterance_lines(t) == '"Ah..." she said.\n"Hey."\n(Me?)\nThen it rained.'


def test_mark_narration_only_long_outside_text():
    long = " ".join(["word"] * NARRATION_MIN_WORDS)
    meta = [{"is_outside_text": True}, {"is_outside_text": True}, {"is_outside_text": False}]
    mark_narration([long, "**BAM**", long], meta)
    assert meta[0].get("mt_narration") and meta[0]["mt_text"] == long
    assert "mt_narration" not in meta[1] and "mt_narration" not in meta[2]


def test_split_symbols():
    assert split_symbols("Ah♡♡ yes", "bold") == [("Ah", "bold"), ("♡♡", "symbol"), (" yes", "bold")]
