#!/usr/bin/env python3
"""Local web UI for translate.py: pick a folder/zip, translate, read the result in the browser.

    uv run python webui.py            # then open http://127.0.0.1:7860
    uv run python webui.py --port 8000

Binds to 127.0.0.1 only. Jobs run one at a time (one GPU) through translate.py.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

import translate as tr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "webui_data"
JOBS_DIR = DATA / "jobs"
DB_PATH = DATA / "jobs.json"
DEFAULT_OUT_ROOT = ROOT / "translated"
INDEX_HTML = ROOT / "webui" / "index.html"
ARCHIVE_EXTS = {".zip", ".cbz"}
LANGUAGES = ["Japanese", "Korean", "Chinese (Simplified)", "Chinese (Traditional)", "English",
             "Spanish", "French", "German", "Portuguese", "Italian", "Russian", "Indonesian",
             "Vietnamese", "Thai", "Polish", "Turkish", "Arabic"]
FINAL = {"done", "failed", "cancelled", "interrupted"}


# --------------------------------------------------------------------------- job store


class Store:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        if DB_PATH.exists():
            try:
                self.jobs = json.loads(DB_PATH.read_text())
            except json.JSONDecodeError:
                self.jobs = {}
        for job in self.jobs.values():  # server restarted mid-job
            if job["status"] in {"running", "queued", "uploading"}:
                job["status"] = "interrupted"
        self.save()

    def save(self) -> None:
        with self.lock:
            DATA.mkdir(parents=True, exist_ok=True)
            tmp = DB_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.jobs, indent=1, ensure_ascii=False))
            tmp.replace(DB_PATH)

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise HTTPException(404, "job not found")
            return job

    def update(self, job_id: str, **fields: Any) -> None:
        with self.lock:
            self.jobs[job_id].update(fields)
            self.save()


store = Store()
work_queue: queue.Queue[str] = queue.Queue()
running: dict[str, subprocess.Popen] = {}


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def worker() -> None:
    while True:
        job_id = work_queue.get()
        try:
            job = store.get(job_id)
        except HTTPException:
            continue
        if job["status"] != "queued":
            continue
        run_job(job_id, job)


def run_job(job_id: str, job: dict[str, Any]) -> None:
    o = job["options"]
    cmd = [sys.executable, str(ROOT / "translate.py"),
           "--input", job["input_dir"], "--output", job["output_dir"],
           "--source", o.get("source") or "Japanese", "--target", o.get("target") or "English",
           "--model", o.get("model") or "26b", "--osb", o.get("osb") or "auto",
           "--context-pages", str(o.get("context_pages", 3))]
    if o.get("font"):
        cmd += ["--font", o["font"]]
    if o.get("instructions"):
        cmd += ["--instructions", o["instructions"]]
    if o.get("offline"):
        cmd += ["--offline"]
    if o.get("adult"):
        cmd += ["--adult"]
    env_extra = {}
    if job.get("resume"):  # only Resume skips pages, and only those this job already wrote
        cmd += ["--skip-existing"]
        env_extra["MT_SKIP_EXISTING_SINCE"] = str(job.get("first_started") or 0)

    Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
    log_path = job_dir(job_id) / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    store.update(job_id, status="running", started=now, first_started=job.get("first_started") or now,
                 finished=None, exit_code=None, error=None)
    with open(log_path, "a") as logf:
        logf.write(f"\n=== run started {time.ctime()} ===\n")
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True,
                                env={**os.environ, "PYTHONUNBUFFERED": "1", **env_extra})
        running[job_id] = proc
        rc = proc.wait()
        running.pop(job_id, None)

    job = store.get(job_id)
    if job.get("cancel_requested"):
        status = "cancelled"
    elif rc == 0:
        status = "done"
    else:
        status = "failed"
    err = None
    if status == "failed":
        tail = [l for l in read_log_tail(log_path, 80) if "[translate]" in l or "rror" in l]
        err = tail[-1] if tail else f"exit code {rc}"
    store.update(job_id, status=status, exit_code=rc, finished=time.time(),
                 cancel_requested=False, error=err)


def read_log_tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 64_000))
        lines = f.read().decode("utf-8", "replace").splitlines()
    return lines[-n:]


# --------------------------------------------------------------------------- pages


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in tr.IMAGE_EXTS


def list_pages(job: dict[str, Any]) -> list[dict[str, Any]]:
    out_dir = Path(job["output_dir"])
    pages = []
    if job["kind"] == "view":
        imgs = sorted((p for p in out_dir.rglob("*") if is_image(p)),
                      key=lambda p: tr.natural_key(p.relative_to(out_dir)))
        for p in imgs:
            rel = p.relative_to(out_dir)
            pages.append({"chapter": str(rel.parent) if str(rel.parent) != "." else "",
                          "name": p.name, "orig": None, "tr": str(rel),
                          "v": int(p.stat().st_mtime)})
        return pages

    in_dir = Path(job["input_dir"])
    if not in_dir.is_dir():
        return []
    # Outputs left in the folder by older runs don't count; only this job's own work does
    since = job.get("first_started") or 0
    for d in tr.image_dirs(in_dir):
        imgs = sorted((p for p in d.iterdir() if is_image(p)), key=lambda p: tr.natural_key(p.name))
        for p in imgs:
            rel = p.relative_to(in_dir)
            out_rel = rel.parent / tr.translated_name(p, "png")
            out_path = out_dir / out_rel
            done = out_path.exists() and out_path.stat().st_mtime >= since
            pages.append({"chapter": str(rel.parent) if str(rel.parent) != "." else "",
                          "name": p.name, "orig": str(rel),
                          "tr": str(out_rel) if done else None,
                          "v": int(out_path.stat().st_mtime) if done else 0})
    return pages


def safe_join(base: Path, rel: str) -> Path:
    p = (base / rel).resolve()
    try:
        p.relative_to(base.resolve())
    except ValueError:
        raise HTTPException(400, "bad path")
    return p


def unwrap(d: Path) -> Path:
    """A folder/zip whose only content is one folder -> use that folder."""
    while True:
        entries = [e for e in d.iterdir() if not e.name.startswith((".", "__MACOSX"))]
        if len(entries) == 1 and entries[0].is_dir():
            d = entries[0]
        else:
            return d


def extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for m in zf.infolist():
            if m.is_dir() or m.filename.startswith("__MACOSX") or Path(m.filename).name.startswith("."):
                continue
            target = safe_join(dest, m.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def prepare_uploaded_input(input_dir: Path) -> Path:
    """Extract uploaded archives in place, then unwrap a single top folder."""
    archives = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in ARCHIVE_EXTS]
    for a in archives:
        target = a.parent if len(archives) == 1 else a.with_suffix("")
        extract_archive(a, target)
        a.unlink()
    return unwrap(input_dir)


def unique_dir(base: Path) -> Path:
    if not base.exists() or not any(base.iterdir()):
        return base
    for i in range(2, 1000):
        cand = base.with_name(f"{base.name} ({i})")
        if not cand.exists():
            return cand
    return base


def clean_name(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", " ", name).strip() or "manga"
    return name[:120]


def job_summary(job: dict[str, Any], with_pages: bool = False) -> dict[str, Any]:
    pages = list_pages(job)
    done = sum(1 for p in pages if p["tr"])
    cover = next((p for p in pages if p["tr"]), None)
    cover_url = None
    if cover:
        cover_url = f"/api/jobs/{job['id']}/img?kind=tr&p={cover['tr']}&v={cover['v']}"
    elif pages and pages[0]["orig"]:
        cover_url = f"/api/jobs/{job['id']}/img?kind=orig&p={pages[0]['orig']}"
    res = {k: job.get(k) for k in ("id", "name", "kind", "status", "input_dir", "output_dir",
                                    "created", "started", "finished", "error", "options")}
    res.update(total=len(pages), done=done, cover=cover_url)
    if with_pages:
        res["pages"] = pages
        log = read_log_tail(job_dir(job["id"]) / "log.txt", 400)
        stage = ""
        for line in reversed(log):
            m = re.search(r"(Processing \d+/\d+: .*|\[translate\] \[\d+/\d+\] .*|llama-server ready|"
                          r"Starting llama-server.*|Downloading .*)", line)
            if m:
                stage = m.group(1).replace("[translate] ", "")
                break
        res["stage"] = stage
        res["log"] = [l for l in log if not l.startswith(("Fontconfig", "  ")) and l.strip()][-60:]
    return res


# --------------------------------------------------------------------------- app

app = FastAPI(title="Manga Translator", docs_url=None, redoc_url=None, openapi_url=None)
ALLOWED_HOSTS: set[str] = set()


@app.middleware("http")
async def local_only(request: Request, call_next):
    # Block DNS-rebinding and cross-site requests: right Host, and a custom header on writes
    if request.headers.get("host", "") not in ALLOWED_HOSTS:
        return Response("forbidden host", status_code=403)
    if request.method not in ("GET", "HEAD") and request.headers.get("x-mt") != "1":
        return Response("missing header", status_code=403)
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text(), headers={"Cache-Control": "no-store"})


@app.get("/api/config")
def config() -> dict[str, Any]:
    fonts = sorted(d.name for d in tr.FONTS_DIR.iterdir() if d.is_dir())
    models = {k: Path(v["gguf"]).exists() for k, v in tr.MODEL_PRESETS.items()}
    return {"languages": LANGUAGES, "fonts": fonts, "models": models,
            "default_out_root": str(DEFAULT_OUT_ROOT), "home": str(Path.home()),
            "osb_ready": tr.OSB_TEXT_MODEL.exists()}


@app.get("/api/browse")
def browse(path: str = "", files: bool = False) -> dict[str, Any]:
    p = Path(path).expanduser() if path else Path.home()
    if not p.exists():
        p = p.parent if p.parent.exists() else Path.home()
    if p.is_file():
        p = p.parent
    p = p.resolve()
    dirs, archives, n_images = [], [], 0
    try:
        for e in sorted(p.iterdir(), key=lambda e: tr.natural_key(e.name)):
            if e.name.startswith("."):
                continue
            try:
                if e.is_dir():
                    dirs.append(e.name)
                elif e.suffix.lower() in ARCHIVE_EXTS and files:
                    archives.append(e.name)
                elif e.suffix.lower() in tr.IMAGE_EXTS:
                    n_images += 1
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(403, f"no permission to read {p}")
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
            "dirs": dirs, "archives": archives, "images": n_images}


@app.get("/api/jobs")
def jobs() -> list[dict[str, Any]]:
    with store.lock:
        items = sorted(store.jobs.values(), key=lambda j: j["created"], reverse=True)
    return [job_summary(j) for j in items]


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> dict[str, Any]:
    return job_summary(store.get(job_id), with_pages=True)


@app.post("/api/jobs")
async def create_job(request: Request) -> dict[str, Any]:
    body = await request.json()
    mode = body.get("mode")  # "upload" | "path" | "view"
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    name = clean_name(body.get("name") or "manga")
    jd = job_dir(job_id)
    job: dict[str, Any] = {"id": job_id, "name": name, "kind": "translate", "created": time.time(),
                           "options": body.get("options") or {}, "uploaded": False,
                           "status": "queued", "error": None}

    if mode == "view":
        p = Path(body.get("path", "")).expanduser()
        if not p.is_dir():
            raise HTTPException(400, "folder not found")
        job.update(kind="view", status="done", input_dir="", output_dir=str(p.resolve()),
                   name=clean_name(body.get("name") or p.name))
    elif mode == "path":
        p = Path(body.get("path", "")).expanduser()
        if p.is_dir():
            input_dir = p.resolve()
        elif p.is_file() and p.suffix.lower() in ARCHIVE_EXTS:
            extract_archive(p, jd / "input")
            input_dir = unwrap(jd / "input")
            job["uploaded"] = True
        else:
            raise HTTPException(400, "pick a folder with images or a .zip/.cbz file")
        job["input_dir"] = str(input_dir)
    elif mode == "upload":
        (jd / "input").mkdir(parents=True, exist_ok=True)
        job.update(input_dir=str(jd / "input"), uploaded=True, status="uploading")
    else:
        raise HTTPException(400, "bad mode")

    if job["kind"] == "translate":
        out = (body.get("output_dir") or "").strip()
        job["output_dir"] = str(Path(out).expanduser().resolve() if out
                                else unique_dir(DEFAULT_OUT_ROOT / name))
        if mode == "path" and not list_pages(job):
            shutil.rmtree(jd, ignore_errors=True)
            raise HTTPException(400, "no images found there")

    with store.lock:
        store.jobs[job_id] = job
        store.save()
    if job["status"] == "queued":
        work_queue.put(job_id)
    return job_summary(job)


@app.put("/api/jobs/{job_id}/upload")
async def upload(job_id: str, path: str, request: Request) -> dict[str, Any]:
    job = store.get(job_id)
    if job["status"] != "uploading":
        raise HTTPException(409, "job is not accepting uploads")
    target = safe_join(Path(job["input_dir"]), path.lstrip("/"))
    if target.suffix.lower() not in tr.IMAGE_EXTS | ARCHIVE_EXTS:
        return {"skipped": path}
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    return {"ok": path}


@app.post("/api/jobs/{job_id}/start")
def start(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job["status"] != "uploading":
        raise HTTPException(409, "already started")
    try:
        input_dir = prepare_uploaded_input(Path(job["input_dir"]))
    except zipfile.BadZipFile:
        store.update(job_id, status="failed", error="not a valid zip/cbz file")
        raise HTTPException(400, "not a valid zip/cbz file")
    store.update(job_id, input_dir=str(input_dir))
    if not list_pages(store.get(job_id)):
        store.update(job_id, status="failed", error="no images found in the upload")
        raise HTTPException(400, "no images found in the upload")
    store.update(job_id, status="queued")
    work_queue.put(job_id)
    return job_summary(store.get(job_id))


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job["status"] in ("queued", "uploading"):
        store.update(job_id, status="cancelled")
    elif job["status"] == "running":
        store.update(job_id, cancel_requested=True)
        proc = running.get(job_id)
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    return job_summary(store.get(job_id))


@app.post("/api/jobs/{job_id}/resume")
def resume(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job["kind"] != "translate" or job["status"] not in FINAL:
        raise HTTPException(409, "cannot resume this job")
    store.update(job_id, status="queued", error=None, resume=True)
    work_queue.put(job_id)
    return job_summary(store.get(job_id))


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job["status"] in ("running", "queued"):
        raise HTTPException(409, "cancel the job first")
    shutil.rmtree(job_dir(job_id), ignore_errors=True)  # uploaded copy + log; output stays
    with store.lock:
        store.jobs.pop(job_id, None)
        store.save()
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/img")
def image(job_id: str, kind: str, p: str) -> FileResponse:
    job = store.get(job_id)
    base = Path(job["output_dir"] if kind == "tr" else job["input_dir"])
    f = safe_join(base, p)
    if not f.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(f, headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    job = store.get(job_id)
    out = Path(job["output_dir"])
    zpath = job_dir(job_id) / "download.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(out.rglob("*"), key=lambda f: tr.natural_key(f.relative_to(out))):
            if is_image(f) or f.name == "glossary.txt":
                zf.write(f, f.relative_to(out))
    return FileResponse(zpath, filename=f"{job['name']} (translated).zip", media_type="application/zip")


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local web UI for the manga translator")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--open", action=argparse.BooleanOptionalAction, default=False,
                    help="Open the UI in the default browser once the server is up")
    a = ap.parse_args()
    ALLOWED_HOSTS.update({f"127.0.0.1:{a.port}", f"localhost:{a.port}"})
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    url = f"http://127.0.0.1:{a.port}"
    print(f"\n  Manga Translator UI:  {url}   (Ctrl+C to stop)\n", flush=True)
    if a.open:
        import webbrowser
        threading.Timer(1.5, webbrowser.open, [url]).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
