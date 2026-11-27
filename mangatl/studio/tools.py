"""The studio's MCP server: what an agent (Claude Code, or any MCP client) sees and does.

Every call is logged to the project's activity (who, which tool, which page, how it went), which
the inspector UI shows live. Tools that change a page take `worker`: Claude Code's subagents share
one MCP connection, so the name is how the studio tells them apart (and keeps two off one page).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mangatl.studio.workspace import Studio, StudioError, StudioProject, encode, fit, overlay, region_view

PLAYBOOK = (Path(__file__).parent / "playbook.md").read_text()

INSTRUCTIONS = """\
Manga translation studio. It detects the text on manga pages, removes it from the art and reads the \
Japanese; you write the English and check how it typesets, page by page, as a coordinator with one \
worker (subagent) per page.

Before anything else call get_playbook and follow it. In short: open_project(folder) -> draft -> \
project_status(wait_seconds=...) until drafted -> read every page and write chapter notes -> one \
worker per page (claim_page, get_notes, get_page, set_translations, view_page, release_page) -> \
review -> consistency pass. Tools that change a page need `worker`: a name for whoever calls."""


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build(studio: Studio) -> MCPServer:
    mcp = MCPServer("mangatl-studio", instructions=INSTRUCTIONS)

    def call(
        tool: str, project: str | None, worker: str, page: str | None, fn: Callable[[StudioProject], Any]
    ) -> Any:
        """Run one tool on a project, logged to its activity; refusals come back as tool errors."""
        p = studio.get(project)
        t0 = time.time()
        try:
            result = fn(p)
        except StudioError as e:
            p.log(worker, tool, page, False, str(e), (time.time() - t0) * 1000)
            raise ToolError(str(e)) from e
        except Exception as e:
            p.log(worker, tool, page, False, f"{type(e).__name__}: {e}", (time.time() - t0) * 1000)
            raise ToolError(f"{type(e).__name__}: {e}") from e
        summary = result.get("summary", "") if isinstance(result, dict) else ""
        p.log(worker, tool, page, True, summary, (time.time() - t0) * 1000)
        return result

    tool = mcp.tool(structured_output=False)

    @tool
    def get_playbook() -> str:
        """The workflow and the translation and typesetting guidelines. Read it first."""
        return PLAYBOOK

    @tool
    def open_project(folder: str, worker: str = "coordinator") -> str:
        """Open a folder of page images (jpg/png/webp, subfolders included) as a project. Returns
        the project id that every other tool takes, and its pages. Opening it again is harmless."""
        try:
            p = studio.open(folder)
        except StudioError as e:
            raise ToolError(str(e)) from e
        p.log(worker, "open_project", None, True, f"{len(p.pages)} pages", 0)
        return _json(p.status())

    @tool
    def list_projects() -> str:
        """The projects the studio has open (from this and earlier sessions)."""
        return _json(
            [
                {"project": p.id, "folder": str(p.folder), "pages": len(p.pages)}
                for p in studio.projects.values()
            ]
        )

    @tool
    def draft(
        project: str, pages: list[str] | None = None, redo: bool = False, worker: str = "coordinator"
    ) -> str:
        """Start drafting pages in the background: detect the text, remove it from the art, read it
        (OCR) and size the lettering, leaving the translation to you. Default: every page not yet
        drafted; `redo` drafts them again (discarding their translations). Follow with
        project_status(wait_seconds=...)."""

        def run(p: StudioProject) -> dict:
            p.start_draft(pages, redo)
            return {**p.status(), "summary": "draft started"}

        return _json(call("draft", project, worker, None, run))

    @tool
    def project_status(project: str, wait_seconds: int = 0) -> str:
        """Each page's draft state, how many regions it has and how many are translated, how many
        have problems, who has it claimed, whether a report is in. `wait_seconds` (up to 600)
        first waits for a running draft to finish."""
        p = studio.get(project)
        p.wait_draft(min(max(wait_seconds, 0), 600))
        return _json(p.status())

    @tool
    def get_page(project: str, page: str, worker: str = "") -> str:
        """A page's regions in reading order: id, kind (bubble / outside = text on the art), box,
        the OCR text, the current translation, and how the translation came out (outcome,
        font_size, max_font, problem). Also the page's problems list."""
        return _json(call("get_page", project, worker, page, lambda p: p.page_info(page)))

    @tool
    def view_page(
        project: str,
        page: str,
        layer: str = "rendered",
        overlay_regions: bool = True,
        max_side: int = 1100,
        worker: str = "",
    ) -> list:
        """The page as an image. layer: "original" (the source), "cleaned" (text removed, nothing
        drawn), "rendered" (with your translations). overlay_regions draws each region's box with
        its reading order and id, coloured by outcome (green fit, orange too small, red does not
        fit, blue untranslated, purple original kept)."""

        def run(p: StudioProject) -> list:
            image = p.image(page, layer)
            if overlay_regions and p.page(page).get("draft") == "drafted":
                size, regions = p.regions_for_overlay(page)
                image = overlay(image, size, regions)
            return [
                f"{page} ({layer})",
                Image(data=encode(fit(image, min(max_side, 1600)), quality=80), format="jpeg"),
            ]

        return call("view_page", project, worker, page, run)

    @tool
    def view_region(project: str, page: str, region: str, worker: str = "") -> list:
        """One region close up: the source (left) beside the rendered result (right), with some
        context around it, plus its data."""

        def run(p: StudioProject) -> list:
            image, data = region_view(p, page, region)
            return [_json(data), Image(data=encode(image), format="jpeg")]

        return call("view_region", project, worker, page, run)

    @tool
    def set_translations(project: str, page: str, translations: dict[str, str], worker: str) -> str:
        """Set translations ({region_id: text}) and re-render the page. Returns each changed region's
        outcome and the page's problems. Text may use \\n for line breaks, *italic*, **bold**,
        ***bold italic***; an empty string leaves the region blank. Claims the page for `worker`."""

        def run(p: StudioProject) -> dict:
            info = p.set_translations(page, translations, worker)
            return {
                **info,
                "summary": f"{len(translations)} regions; {len(info['problems'])} problems on the page",
            }

        return _json(call("set_translations", project, worker, page, run))

    @tool
    def claim_page(project: str, page: str, worker: str, take_over: bool = False) -> str:
        """Claim a page for `worker`, so no one else changes it (lapses after 15 minutes without a
        call). take_over: replace another worker's claim; only once that worker has returned (the
        coordinator, after a worker ended without releasing its page)."""

        def run(p: StudioProject) -> dict:
            previous = p.claim(page, worker, take_over)
            note = f"taken over from {previous}" if previous else "claimed"
            return {"page": page, "claimed_by": worker, "summary": note}

        return _json(call("claim_page", project, worker, page, run))

    @tool
    def release_page(project: str, page: str, worker: str, report: str) -> str:
        """Release a page when done, with a short report: decisions made and problems left (missed
        text, bad cleaning, unreadable OCR, wrong reading order). The report is kept with the page."""

        def run(p: StudioProject) -> dict:
            p.release(page, worker, report)
            return {"page": page, "released": True, "summary": report[:200]}

        return _json(call("release_page", project, worker, page, run))

    @tool
    def get_notes(project: str, worker: str = "") -> str:
        """The chapter notes every worker follows: characters, names, terms, tone, honorific policy."""
        return call("get_notes", project, worker, None, lambda p: p.notes() or "(no notes yet)")

    @tool
    def update_notes(project: str, text: str, mode: str = "replace", worker: str = "coordinator") -> str:
        """Write the chapter notes (markdown). mode: "replace" or "append"."""

        def run(p: StudioProject) -> dict:
            new = p.update_notes(text, mode)
            return {"chars": len(new), "summary": f"notes {mode}d ({len(new)} chars)"}

        return _json(call("update_notes", project, worker, None, run))

    @mcp.prompt()
    def translate_folder(folder: str) -> str:
        """Translate a folder of manga pages with the studio."""
        return f"Translate the manga pages in {folder} with the studio's tools, following this playbook.\n\n{PLAYBOOK}"

    return mcp
