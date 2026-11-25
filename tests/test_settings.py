"""Settings model and presets."""

import json

import pytest
from pydantic import ValidationError

from mangatl import presets
from mangatl.settings import Settings


def test_round_trip_and_defaults():
    s = Settings()
    again = Settings.model_validate_json(s.to_json())
    assert again.model_dump() == s.model_dump()
    assert s.typesetting.sfx == "label" and s.translation.vision == "short"


def test_secrets_never_serialised(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "SECRET_HF")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "SECRET_KEY")
    s = Settings().resolve_secrets()
    assert s.runtime.hf_token == "SECRET_HF" and s.translation.api_key == "SECRET_KEY"
    for text in (s.to_json(), repr(s)):
        assert "SECRET" not in text
    data = json.loads(s.to_json())
    assert "hf_token" not in data["runtime"] and "api_key" not in data["translation"]


def test_validation_and_updates():
    s = Settings()
    with pytest.raises(ValidationError):
        s.translation.top_k = 999
    with pytest.raises(ValidationError):
        Settings.model_validate({"translation": {"nonsense": 1}})
    u = s.updated(**{"typesetting.sfx": "redraw"})
    assert u.typesetting.sfx == "redraw" and s.typesetting.sfx == "label"
    with pytest.raises(KeyError):
        s.updated(**{"typesetting.nope": 1})
    v = presets.apply_overrides(s, ["translation.temperature=0", "typesetting.bubbles.font=Bangers"])
    assert v.translation.temperature == 0.0 and isinstance(v.translation.temperature, float)
    assert v.typesetting.bubbles.font == "Bangers"


def build(**kw):
    kw.setdefault("batch", True)
    kw.setdefault("outside_text", True)
    return presets.build(output=presets.ROOT / "out", log=lambda m: None, **kw)


def test_presets_batch_vs_single():
    batch, single = build(), build(batch=False)
    assert batch.translation.glossary and batch.translation.context_pages == 3
    assert batch.translation.glossary_path == presets.ROOT / "out" / "glossary.txt"
    assert not single.translation.glossary and single.translation.context_pages == 0
    assert "new-terms block" in batch.translation.special_instructions
    assert "new-terms block" not in single.translation.special_instructions


def test_presets_source_language_and_vision():
    ko = build(source="Korean")
    assert ko.ocr.method == "paddleocr-vl-1.6" and ko.translation.reading_direction == "ltr"
    assert build(model="12b").translation.vision == "off"  # doesn't fit next to the 12B
    assert build(llm_url="http://x/v1").translation.vision == "off"
    assert build(llm_url="http://x/v1").translation.model == "default"
    assert build(vision="full").translation.vision == "full"
    assert not build(outside_text=False).detection.outside_text.cover_with_bubbles


def test_honorifics_note():
    assert "Drop Japanese honorifics" in build().translation.special_instructions
    assert "Keep Japanese honorifics" in build(honorifics="keep").translation.special_instructions
