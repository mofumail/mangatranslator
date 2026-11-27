"""The studio over MCP, end to end on one om10 page: draft it, write the harness run's
translations through set_translations, and get exactly the harness run's page.

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
    stored = Project(harness / "project.mtl", create=False).load_page(PAGE)
    translations = {r.id: r.translation or "" for r in stored.regions if r.order is not None}
    engine = Engine()
    studio = Studio(engine, tmp_path / "ws")
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
            await call("release_page", project=pid, page=PAGE, worker="page-002", report="done")
            return info

    try:
        info = asyncio.run(run())
    finally:
        engine.close()
    assert info["translated"] == info["regions_total"] == len(translations)
    assert all(r["outcome"] in ("fit", "kept_original") for r in info["regions"]), info["regions"]
    project = next(iter(studio.projects.values()))
    assert project.meta["pages"][PAGE]["report"]["text"] == "done"
    assert [a["tool"] for a in project.activity][:2] == ["open_project", "get_page"]
    a = np.asarray(Image.open(project.output_path(PAGE)))
    b = np.asarray(Image.open(harness / f"{Path(PAGE).stem}_translated.png"))
    assert a.shape == b.shape and np.array_equal(a, b)
