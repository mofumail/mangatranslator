"""The studio's pieces that need no models: what counts as a problem, overlays, project ids."""

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from mangatl.studio.workspace import _problem, _slug, overlay  # noqa: E402


def region(**kw):
    return {
        "id": "r1",
        "order": 1,
        "bbox": [10, 10, 60, 90],
        "translation": "Hi.",
        "outcome": "fit",
        "font_size": 20,
        "max_font": 20,
        **kw,
    }


def test_problems_an_agent_must_fix():
    assert _problem(region()) is None
    assert _problem(region(translation="")) == "not translated"
    assert "minimum font size" in str(_problem(region(outcome="below_min_font")))
    assert "Japanese lettering was put back" in str(_problem(region(outcome="kept_original_no_fit")))
    assert _problem(region(outcome="kept_original")) is None  # on purpose (a labelled SFX)
    assert str(_problem(region(font_size=10))).startswith("small text")


def test_overlay_scales_boxes_from_the_work_image():
    image = Image.new("RGB", (200, 300), "white")
    out = overlay(image, (100, 150), [region()])  # the work image was half the size
    assert out.size == (200, 300)
    assert out.getpixel((20, 100)) != (255, 255, 255)  # the box's left edge, at x = 10 * 2


def test_project_ids_are_stable_and_distinct():
    assert _slug(Path("/a/My Manga ch.1")) == _slug(Path("/a/My Manga ch.1"))
    assert _slug(Path("/a/ch1")) != _slug(Path("/b/ch1"))
    assert _slug(Path("/a/My Manga ch.1")).startswith("My-Manga-ch-1-")
