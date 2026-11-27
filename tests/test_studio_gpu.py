"""The studio over MCP, end to end on one om10 page: draft it, write the harness run's
translations through set_translations, and get exactly the harness run's page (with the studio's
lettering defaults off); then letter regions with styles and previews.

Needs the models, a GPU and a local harness run of om10 (see test_stages_gpu); skipped otherwise.
"""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("mcp")

from tests.test_stages_gpu import _source_project  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGE = "002.jpg"


def test_an_agent_translates_a_page_over_mcp(tmp_path):
    import numpy as np
    from mcp import Client
    from PIL import Image

    from mangatl.engine import Engine
    from mangatl.project.store import Project
    from mangatl.studio import tools
    from mangatl.studio.workspace import Studio

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    src = _source_project()
    if src is None:
        pytest.skip("no stored om10 project (run eval/harness.py check --only om10)")
        raise AssertionError
    harness = src.parent
    expected = np.asarray(Image.open(harness / f"{Path(PAGE).stem}_translated.png"))
    stored = Project(harness / "project.mtl", create=False).load_page(PAGE)
    translations = {r.id: r.translation or "" for r in stored.regions if r.order is not None}
    engine = Engine()
    # the pipeline's lettering, so the page can match the harness run
    plain = ["typesetting.presets={}", "typesetting.em_dash=false", "typesetting.synthetic_styles=false"]
    studio = Studio(engine, tmp_path / "ws", plain)
    server = tools.build(studio)

    async def run() -> dict:
        async with Client(server) as c:

            async def call(name: str, **args):
                result = await c.call_tool(name, args)
                text = next((x.text for x in result.content if x.type == "text"), "")
                return result, text

            _, text = await call("open_project", folder=str(ROOT / "eval/in/tojime_no_siora"))
            pid = json.loads(text)["project"]
            result, _ = await call("get_page", project=pid, page=PAGE)
            assert result.is_error  # not drafted yet
            await call("draft", project=pid, pages=[PAGE])
            _, text = await call("project_status", project=pid, wait_seconds=600)
            assert next(p for p in json.loads(text)["pages"] if p["page"] == PAGE)["draft"] == "drafted"

            await call("claim_page", project=pid, page=PAGE, worker="page-002")
            result, _ = await call(
                "set_translations", project=pid, page=PAGE, translations={}, worker="intruder"
            )
            assert result.is_error  # claimed by someone else
            _, text = await call(
                "set_translations", project=pid, page=PAGE, translations=translations, worker="page-002"
            )
            info = json.loads(text)
            result, _ = await call("view_page", project=pid, page=PAGE)
            assert [x.type for x in result.content] == ["text", "image"]
            result, _ = await call("view_region", project=pid, page=PAGE, region=next(iter(translations)))
            assert [x.type for x in result.content] == ["text", "image"]
            out = Path(json.loads((await call("project_status", project=pid))[1])["output"])
            same = np.array_equal(np.asarray(Image.open(out / "002_translated.png")), expected)

            # lettering: a sheet, a style, a preview
            _, text = await call(
                "set_style_sheet", project=pid, presets={"shout": {"bold": True, "scale": 1.3}}
            )
            assert "shout" in json.loads(text)["presets"]
            rid = next(iter(translations))
            result, text = await call(
                "set_style",
                project=pid,
                page=PAGE,
                styles={rid: {"preset": "shout", "align": "left"}},
                worker="page-002",
            )
            assert not result.is_error, text
            styled = json.loads(text)["regions"][0]
            assert styled["preset"] == "shout" and styled["style"]["align"] == "left"
            assert {"fill", "margin", "lines"} <= set(styled)
            result, _ = await call(
                "set_style", project=pid, page=PAGE, styles={rid: {"font": "Nope"}}, worker="page-002"
            )
            assert result.is_error  # no such font
            result, text = await call(
                "preview_region",
                project=pid,
                page=PAGE,
                region=rid,
                variants=[{"style": {"scale": 1.5}}, {"text": "Hm."}],
            )
            assert [x.type for x in result.content] == ["text", "image"] and len(json.loads(text)) == 2
            _, text = await call("get_page", project=pid, page=PAGE)
            assert next(r for r in json.loads(text)["regions"] if r["id"] == rid)["style"] == {
                "preset": "shout",
                "align": "left",
            }
            await call("release_page", project=pid, page=PAGE, worker="page-002", report="done")
            return {**info, "same_as_harness": same}

    try:
        info = asyncio.run(run())
    finally:
        engine.close()
    assert info["translated"] == info["regions_total"] == len(translations)
    assert all(r["outcome"] in ("fit", "kept_original") for r in info["regions"]), info["regions"]
    project = next(iter(studio.projects.values()))
    assert project.meta["pages"][PAGE]["report"]["text"] == "done"
    assert [a["tool"] for a in project.activity][:2] == ["open_project", "get_page"]
    assert info["same_as_harness"]
