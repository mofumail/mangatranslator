"""The studio's HTTP side: the MCP endpoint (/mcp) and the inspector UI (/ and /api), one app.

The UI is read-only for now: projects, pages (original / cleaned / rendered, region overlays),
each region's text and outcome, worker reports, the chapter notes and the live activity feed.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from starlette.applications import Starlette

from mangatl.studio import tools
from mangatl.studio.workspace import Studio, StudioError, encode, overlay

STATIC = Path(__file__).parent / "static"


class OpenRequest(BaseModel):
    folder: str


def build_app(studio: Studio) -> Starlette:
    mcp = tools.build(studio)
    app = mcp.streamable_http_app(streamable_http_path="/mcp")  # runs the MCP session manager
    ui = FastAPI(title="mangatl studio", docs_url=None, redoc_url=None)

    def project(pid: str):
        try:
            return studio.get(pid)
        except StudioError as e:
            raise HTTPException(404, str(e)) from e

    @ui.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @ui.get("/api/state")
    def state():
        status = studio.engine.status()
        return {
            "projects": [
                {"id": p.id, "folder": str(p.folder), "pages": len(p.pages), "output": str(p.out)}
                for p in studio.projects.values()
            ],
            "engine": {
                k: status.get(k)
                for k in ("device", "free_mb", "total_mb", "allocated_mb", "loads", "unloads")
            },
            "loaded": sorted(status.get("models", {})),
        }

    @ui.post("/api/projects")
    def open_project(req: OpenRequest):
        try:
            p = studio.open(req.folder)
        except StudioError as e:
            raise HTTPException(400, str(e)) from e
        return {"id": p.id}

    @ui.get("/api/projects/{pid}")
    def project_status(pid: str):
        p = project(pid)
        status = p.status()
        for entry in status["pages"]:
            entry["version"] = p.meta["pages"][entry["page"]].get("version", 0)
        return {**status, "notes": p.notes()}

    @ui.get("/api/projects/{pid}/page")
    def page(pid: str, page: str = Query(...)):
        p = project(pid)
        meta = p.page(page)
        info = p.page_info(page) if meta.get("draft") == "drafted" else {"page": page, "regions": []}
        return {
            **info,
            "draft": meta.get("draft"),
            "report": meta.get("report"),
            "version": meta.get("version", 0),
        }

    @ui.get("/api/projects/{pid}/image")
    def image(pid: str, page: str = Query(...), layer: str = "rendered", regions: bool = False):
        p = project(pid)
        try:
            im = p.image(page, layer)
            if regions and p.page(page).get("draft") == "drafted":
                size, items = p.regions_for_overlay(page)
                im = overlay(im, size, items)
        except StudioError as e:
            raise HTTPException(400, str(e)) from e
        return Response(
            encode(im, quality=92), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"}
        )

    @ui.get("/api/projects/{pid}/activity")
    def activity(pid: str, after: int = 0):
        p = project(pid)
        with p.lock:
            return [a for a in p.activity if a["seq"] > after][-500:]

    app.mount("/", ui)  # after /mcp, so the MCP route matches first
    return app
