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
from mcp_types import ToolAnnotations

from mangatl.studio.workspace import Studio, StudioError, StudioProject, encode, fit, overlay, region_view

PLAYBOOK = (Path(__file__).parent / "playbook.md").read_text()

INSTRUCTIONS = """\
Manga translation studio. It detects the text on manga pages, removes it from the art and reads the \
Japanese; you write the English and letter it, page by page, as a coordinator with one worker \
(subagent) per page.

Before anything else call get_playbook and follow it. In short: open_project(folder) -> draft -> \
project_status(wait_seconds=60) until drafted -> read the chapter (get_script), set the glossary, \
notes and style sheet -> one worker per page (claim_page, get_page, annotate_regions, \
set_translations with speakers and styles, view_page, preview_region, release_page) -> review_page \
-> check_consistency -> export_project. Tools that change a page need `worker`: a name for whoever \
calls."""


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
    read = mcp.tool(structured_output=False, annotations=ToolAnnotations(read_only_hint=True))

    @read
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

    @read
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

    @read
    def project_status(project: str, wait_seconds: int = 0) -> str:
        """Each page's draft state, how many regions it has and how many are translated, how many
        have problems, who has it claimed, whether a report is in. `wait_seconds` (up to 600)
        first waits for a running draft to finish."""
        p = studio.get(project)
        p.wait_draft(min(max(wait_seconds, 0), 600))
        return _json(p.status())

    @read
    def get_page(project: str, page: str, worker: str = "") -> str:
        """A page's regions in reading order: id, kind (bubble / outside = text on the art), box,
        the OCR text, the current translation, and how the translation came out (outcome,
        font_size, max_font, problem). Also the page's problems list."""
        return _json(call("get_page", project, worker, page, lambda p: p.page_info(page)))

    @read
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

    @read
    def view_region(
        project: str,
        page: str,
        region: str,
        layers: list[str] | None = None,
        zoom: float = 1.0,
        boxes: bool = False,
        worker: str = "",
    ) -> list:
        """One region close up, with some context, in each of `layers` side by side ("original",
        "cleaned", "rendered"; default original and rendered), plus its data. zoom: 0.5-3 (bigger
        image); boxes: draw the region's box (blue) and where its text landed (green)."""

        def run(p: StudioProject) -> list:
            image, data = region_view(p, page, region, tuple(layers or ("original", "rendered")), zoom, boxes)
            return [_json(data), Image(data=encode(image), format="jpeg")]

        return call("view_region", project, worker, page, run)

    @tool
    def set_translations(
        project: str,
        page: str,
        translations: dict[str, str],
        worker: str,
        styles: dict[str, Any] | None = None,
        speakers: dict[str, str] | None = None,
    ) -> str:
        """Set translations ({region_id: text}) and re-render the page. Returns each changed region's
        outcome and measurements, and the page's problems. Text may use \\n for line breaks,
        *italic*, **bold**, ***bold italic***; an empty string leaves the region blank. `styles`:
        lettering per region in the same call (see set_style); `speakers`: {region_id: who says it}.
        Doesn't claim the page (claim_page does), but refuses a page another worker holds."""

        def run(p: StudioProject) -> dict:
            info = p.set_translations(page, translations, worker, styles, speakers)
            n = len(set(translations) | set(styles or {}) | set(speakers or {}))
            return {**info, "summary": f"{n} regions; {len(info['problems'])} problems on the page"}

        return _json(call("set_translations", project, worker, page, run))

    @tool
    def set_style(project: str, page: str, styles: dict[str, Any], worker: str) -> str:
        """Letter regions beyond the automatic fit, {region_id: style} (merged into the region's
        current style; a field set to null clears it, null for the region clears its style), and
        re-render. Style fields, all optional:
          preset: a preset of the style sheet (get_style_sheet: dialogue, shout, thought, whisper,
            narration, ...); the other fields override it
          font: a font pack (list_fonts); caps: capitals; bold / italic: the whole text
          size: the largest size in px the text is fitted at, or scale: that size relative to the
            page's dialogue size (1.2 = 20% bigger)
          align: center | left | right; line_spacing: multiple of the normal (1.0)
          rotation: degrees, clockwise (-90..90)
          color / outline_color: "#rrggbb"; outline: outline width in px (0 none)
          box: [x1, y1, x2, y2] the text area on the page instead of the bubble's
          offset: [dx, dy] move the text block in px; padding: margin to the bubble's edge in px
        Returns each region's outcome and measurements (fill: share of the bubble the text covers;
        margin: px to the bubble's edge; lines)."""

        def run(p: StudioProject) -> dict:
            info = p.set_style(page, styles, worker)
            return {
                **info,
                "summary": f"styled {len(styles)} regions; {len(info['problems'])} problems on the page",
            }

        return _json(call("set_style", project, worker, page, run))

    @read
    def preview_region(
        project: str, page: str, region: str, variants: list[dict[str, Any]], worker: str = ""
    ) -> list:
        """Try up to 4 variants of one region side by side without changing anything. Each variant
        is {"text": ..., "style": {...}} (both optional; the style merges into the region's own, as
        in set_style). Returns the image (variants numbered left to right) and each variant's
        outcome and measurements. Apply the one you pick with set_translations / set_style."""

        def run(p: StudioProject) -> list:
            image, results = p.preview(page, region, variants)
            return [_json(results), Image(data=encode(image), format="jpeg")]

        return call("preview_region", project, worker, page, run)

    @read
    def list_fonts(worker: str = "") -> list:
        """The font packs a style can use, with the styles each has (bold / italic are drawn from
        the regular when a pack lacks them) and a sample line of each."""
        from mangatl.settings import FONTS_DIR
        from mangatl.studio.workspace import font_samples
        from mangatl.typeset.style import font_packs

        return [_json(font_packs(FONTS_DIR)), Image(data=encode(font_samples()), format="jpeg")]

    @read
    def get_style_sheet(project: str, worker: str = "") -> str:
        """The chapter's style sheet: its presets (named lettering regions refer to) and base
        dialogue size, and the fonts available."""
        return _json(call("get_style_sheet", project, worker, None, lambda p: p.style_sheet()))

    @tool
    def set_style_sheet(
        project: str,
        presets: dict[str, Any] | None = None,
        base_size: float | None = None,
        replace: bool = False,
        worker: str = "coordinator",
    ) -> str:
        """Change the chapter's style sheet and re-render every drafted page with it. presets:
        {name: style fields (as set_style, without preset / box / offset)}, merged into the sheet
        (null removes a preset; replace=true: only these). "dialogue" applies to every bubble and
        "outside" to text on the art unless a region names another preset. base_size: one dialogue
        size (px) for the whole chapter instead of each page's own (0: back to per page)."""

        def run(p: StudioProject) -> dict:
            result = p.set_style_sheet(presets, base_size, replace, worker)
            return {**result, "summary": f"sheet set; {sum(result['problems_per_page'].values())} problems"}

        return _json(call("set_style_sheet", project, worker, None, run))

    @tool
    def annotate_regions(project: str, page: str, regions: dict[str, Any], worker: str) -> str:
        """Record what you know about regions, {region_id: {fields}}, any of:
          speaker: who says it (one consistent name per character; "" clears)
          source: the Japanese as actually written, when the OCR misread it
          dismissed: why it isn't text at all (a false detection: speed lines, art); the region is
            drawn as the original art and needs no translation. "" restores it.
          link: a name shared by the bubbles of one sentence split across bubbles ("" unlinks);
            translate linked bubbles as one unit
        Re-renders the page and returns the regions."""

        def run(p: StudioProject) -> dict:
            info = p.annotate(page, regions, worker)
            return {**info, "summary": f"annotated {len(regions)} regions"}

        return _json(call("annotate_regions", project, worker, page, run))

    @tool
    def set_order(project: str, page: str, order: list[str], worker: str) -> str:
        """Fix the reading order: every translatable region id of the page, once, in the order they
        are read. Returns the page."""

        def run(p: StudioProject) -> dict:
            info = p.set_order(page, order, worker)
            return {**info, "summary": "reading order set"}

        return _json(call("set_order", project, worker, page, run))

    @read
    def get_script(
        project: str, pages: list[str] | None = None, speaker: str | None = None, worker: str = ""
    ) -> str:
        """The chapter (or some pages) as a script: per page, each region in reading order with
        its speaker, source and translation, and its problem if any. `speaker`: only that
        character's lines. For reading the chapter, reviewing and the consistency pass."""
        return call("get_script", project, worker, None, lambda p: p.script(pages, speaker))

    @read
    def get_context(project: str, page: str, worker: str = "") -> str:
        """The pages before and after this one as script (their latest translations, or the source
        while untranslated): how the conversation arrives and leaves."""
        return call("get_context", project, worker, page, lambda p: p.context(page))

    @read
    def get_glossary(project: str, worker: str = "") -> str:
        """The chapter's glossary: names and terms with their chosen English."""
        return _json(call("get_glossary", project, worker, None, lambda p: p.glossary()))

    @tool
    def set_glossary(
        project: str, entries: list[dict[str, Any]], replace: bool = False, worker: str = "coordinator"
    ) -> str:
        """Add or change glossary entries: [{"source": Japanese, "english": chosen English,
        "kind": "name" | "term" | ..., "note": ...}], by source term; "english": "" removes one.
        replace: the glossary becomes exactly these."""

        def run(p: StudioProject) -> dict:
            g = p.set_glossary(entries, replace)
            return {"glossary": g, "summary": f"{len(g)} entries"}

        return _json(call("set_glossary", project, worker, None, run))

    @read
    def check_consistency(project: str, worker: str = "") -> str:
        """The consistency pass's findings: glossary terms whose English is missing where the source
        has them, variants of a chosen English (e.g. plural or misspelt names), honorifics left in
        the translations, how many lines each speaker has and bubbles without a speaker, open
        problems."""
        return _json(call("check_consistency", project, worker, None, lambda p: p.check_consistency()))

    @read
    def region_history(project: str, page: str, region: str, worker: str = "") -> str:
        """A region's earlier translations, source reads, speakers, styles and dismissals, oldest
        first, with who made each change. To go back, set the old value again."""
        return _json(call("region_history", project, worker, page, lambda p: p.history(page, region)))

    @tool
    def review_page(
        project: str, page: str, verdict: str, notes: str = "", worker: str = "coordinator"
    ) -> str:
        """The coordinator's verdict on a page: "approved", or "changes_requested" with notes saying
        what to change. The notes appear in get_page (reviews) for the next worker on the page."""

        def run(p: StudioProject) -> dict:
            return {**p.review(page, worker, verdict, notes), "summary": f"{verdict}: {notes[:150]}"}

        return _json(call("review_page", project, worker, page, run))

    @tool
    def export_project(project: str, cbz: bool = False, worker: str = "coordinator") -> str:
        """Finish: write the chapter script (script.md) and, with cbz, a .cbz of the rendered
        pages; returns where they are and what is still unresolved (problems, pages not approved,
        claims)."""

        def run(p: StudioProject) -> dict:
            out = p.export(cbz)
            return {**out, "summary": f"exported; {len(out['unresolved'])} unresolved"}

        return _json(call("export_project", project, worker, None, run))

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

    @read
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
