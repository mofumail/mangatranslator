"""The studio's state: one engine, the open projects, and per project its pages, claims, notes,
reports and activity. The MCP tools (mangatl.studio.mcp) and the inspector UI
(mangatl.studio.web) are two views of this.

A project is a folder of page images. Its work lives in the studio's workspace, one directory
per project:

    project.mtl     the page documents (mangatl.project.store), with their history
    out/            the rendered pages (<name>_translated.png)
    studio.json     per page: draft status, last render's outcomes, the worker's report
    notes.md        the chapter notes the agents share (characters, terms, style decisions)
    activity.jsonl  every tool call: who, what, on which page, how it went

Drafting runs the pipeline up to the translation (detect, clean, OCR, font sizes) and leaves
the text to the agent; set_translations draws it and measures how each region came out.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from mangatl.common.log import log_message
from mangatl.engine import Engine, Priority
from mangatl.engine.models import Model
from mangatl.settings import FONTS_DIR, Settings
from mangatl.typeset.style import LetterStyle, font_packs, resolve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
WORK_CACHE = 8  # stored pages kept loaded (a page's working data: masks, crops, canvas)
CLAIM_TTL_S = 15 * 60  # a claim nobody renews lapses
ACTIVITY_KEPT = 5000  # in memory, for the UI (all of it stays in activity.jsonl)
SMALL_TEXT = 0.75  # font size under this share of the region's size cap reads as crowded
UNDERFILLED = 0.2  # text at its size cap covering less of the bubble than this: a hint to go bigger
MAX_PREVIEWS = 4

# The style sheet new projects start with: lettering in capitals, as in English manga
DEFAULT_PRESETS = {
    "dialogue": {"font": "CC Wild Words", "caps": True},
    "shout": {"font": "CC Wild Words", "caps": True, "bold": True, "scale": 1.3},
    "thought": {"font": "CC Wild Words", "caps": True, "italic": True},
    "whisper": {"font": "CC Wild Words", "caps": True, "scale": 0.85},
    "narration": {"font": "Anime Ace", "caps": True},
}

# How a region's text came out (stages.render outcomes); the ones that need work
PROBLEMS = {
    "below_min_font": "text only fits below the minimum font size (too long for the bubble)",
    "kept_original_no_fit": "text does not fit at all; the Japanese lettering was put back",
    "failed": "text could not be placed",
}


class StudioError(ValueError):
    """A request the studio refuses, with a message for the agent."""


def _slug(folder: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", folder.name).strip("-") or "project"
    return f"{name}-{hashlib.sha1(str(folder).encode()).hexdigest()[:6]}"


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


class Studio:
    def __init__(self, engine: Engine, workspace: Path, overrides: list[str] | None = None):
        self.engine = engine
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.overrides = overrides or []
        self.lock = threading.Lock()
        self.projects: dict[str, StudioProject] = {}
        for meta in sorted(self.workspace.glob("*/studio.json")):
            try:
                folder = Path(json.loads(meta.read_text())["folder"])
                self.projects[meta.parent.name] = StudioProject(self, meta.parent, folder)
            except Exception as e:  # a broken project shouldn't keep the studio from starting
                log_message(f"Studio: skipped {meta.parent.name}: {e!r}", always_print=True)

    def settings(self, project_dir: Path) -> Settings:
        """The settings drafts are made with: translate.py's defaults, without the parts that
        only serve the LLM translation, and the pipeline's stages in plain order."""
        from mangatl import presets

        outside = self.engine.path(Model.OUTSIDE_TEXT).exists()
        settings = presets.build(batch=True, output=project_dir, outside_text=outside, log=lambda *_: None)
        settings = settings.updated(
            **{
                "output.project": project_dir / "project.mtl",
                "runtime.overlap_llm_with_inpaint": False,
                "runtime.verbose": False,
                "preprocessing.split_long_strips": False,
                "translation.full_page_context": False,
                "typesetting.presets": {k: LetterStyle.model_validate(v) for k, v in DEFAULT_PRESETS.items()},
                "typesetting.synthetic_styles": True,
                "typesetting.em_dash": True,
            }
        )
        return presets.apply_overrides(settings, self.overrides) if self.overrides else settings

    def open(self, folder: str | Path) -> StudioProject:
        path = Path(folder).expanduser().resolve()
        if not path.is_dir():
            raise StudioError(f"not a folder: {path}")
        with self.lock:
            for p in self.projects.values():
                if p.folder == path:
                    p.refresh_pages()
                    return p
            project = StudioProject(self, self.workspace / _slug(path), path)
            self.projects[project.id] = project
            return project

    def get(self, project_id: str | None) -> StudioProject:
        if project_id is None:
            if len(self.projects) == 1:
                return next(iter(self.projects.values()))
            raise StudioError(f"name the project; open ones: {sorted(self.projects)}")
        if project_id not in self.projects:
            raise StudioError(
                f"no open project {project_id!r}; open ones: {sorted(self.projects)} (open_project first)"
            )
        return self.projects[project_id]


class StudioProject:
    def __init__(self, studio: Studio, directory: Path, folder: Path):
        from mangatl.project import capture

        self.studio = studio
        self.engine = studio.engine
        self.dir = directory
        self.id = directory.name
        self.folder = folder
        self.out = directory / "out"
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()  # meta, claims, activity, notes
        self._page_locks: dict[str, threading.RLock] = defaultdict(threading.RLock)
        self._work: OrderedDict[str, Any] = OrderedDict()  # page id -> WorkPage
        self._work_lock = threading.Lock()  # the cache (a page's own use is under its page lock)
        self.claims: dict[str, tuple[str, float]] = {}
        self.job = None
        meta_path = directory / "studio.json"
        self.meta: dict[str, Any] = (
            json.loads(meta_path.read_text())
            if meta_path.exists()
            else {"folder": str(folder), "created": time.time(), "pages": {}}
        )
        for page in self.meta["pages"].values():
            if page.get("draft") in ("queued", "drafting"):  # a draft the last studio didn't finish
                page["draft"] = "pending"
            if "report" in page:  # one report per page, before reports accumulated
                page.setdefault("reports", []).append(page.pop("report"))
        self.activity: list[dict] = []
        log = directory / "activity.jsonl"
        if log.exists():
            self.activity = [
                json.loads(line) for line in log.read_text().splitlines()[-ACTIVITY_KEPT:] if line
            ]
        self.config = self._with_sheet(studio.settings(directory))
        self.store = capture.open_project(directory / "project.mtl")
        self.refresh_pages()

    # ------------------------------------------------------------------ pages
    def refresh_pages(self) -> None:
        from mangatl.pipeline.batch import natural_path_sort_key

        files = sorted(
            (p for p in self.folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
            key=natural_path_sort_key,
        )
        with self.lock:
            self.pages = [p.relative_to(self.folder).as_posix() for p in files]
            for pid in self.pages:
                self.meta["pages"].setdefault(pid, {"draft": "pending"})
            self._save_meta()

    def page(self, page_id: str) -> dict:
        if page_id not in self.meta["pages"] or page_id not in self.pages:
            raise StudioError(f"no page {page_id!r} in {self.id}; pages: {self.pages}")
        return self.meta["pages"][page_id]

    def _drafted(self, page_id: str) -> dict:
        page = self.page(page_id)
        if page.get("draft") != "drafted":
            raise StudioError(f"page {page_id} isn't drafted yet ({page.get('draft')}); draft it first")
        return page

    def output_path(self, page_id: str) -> Path:
        p = Path(page_id)
        return self.out / p.parent / f"{p.stem}_translated.png"

    def _save_meta(self) -> None:
        with self.lock:
            _write_json(self.dir / "studio.json", self.meta)

    # ------------------------------------------------------------------ activity, claims, notes
    def log(self, worker: str, tool: str, page: str | None, ok: bool, summary: str, ms: float) -> None:
        with self.lock:
            entry = {
                "seq": (self.activity[-1]["seq"] + 1) if self.activity else 1,
                "t": time.time(),
                "worker": worker or "?",
                "tool": tool,
                "page": page,
                "ok": ok,
                "summary": summary[:500],
                "ms": round(ms),
            }
            self.activity.append(entry)
            del self.activity[:-ACTIVITY_KEPT]
            with open(self.dir / "activity.jsonl", "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def claimed_by(self, page_id: str) -> str | None:
        claim = self.claims.get(page_id)
        if claim and time.time() - claim[1] < CLAIM_TTL_S:
            return claim[0]
        return None

    def claim(self, page_id: str, worker: str, take_over: bool = False) -> str | None:
        """Claim a page for `worker`; returns the holder it took over from (take_over), if any."""
        if not worker:
            raise StudioError("say who you are: pass worker (e.g. 'page-3')")
        self.page(page_id)
        with self.lock:
            holder = self.claimed_by(page_id)
            if holder and holder != worker and not take_over:
                left = CLAIM_TTL_S - (time.time() - self.claims[page_id][1])
                raise StudioError(
                    f"page {page_id} is claimed by {holder} (lapses in {left:.0f} s unless renewed; "
                    "claim_page with take_over=true once that worker has finished)"
                )
            self.claims[page_id] = (worker, time.time())
            page = self.page(page_id)
            if page.get("state") in (None, "drafted", "changes_requested"):
                page["state"] = "in_progress"
                self._save_meta()
            return holder if holder != worker else None

    def _may_edit(self, page_id: str, worker: str) -> None:
        """Anyone may change a page nobody holds (a coordinator's touch-up doesn't claim it);
        a page held by another worker is theirs until released or taken over."""
        if not worker:
            raise StudioError("say who you are: pass worker (e.g. 'page-3')")
        holder = self.claimed_by(page_id)
        if holder and holder != worker:
            raise StudioError(
                f"page {page_id} is claimed by {holder}; claim_page with take_over=true once that worker has finished"
            )
        if holder == worker:
            with self.lock:
                self.claims[page_id] = (worker, time.time())  # renewed by use

    def release(self, page_id: str, worker: str, report: str) -> None:
        self.claim(page_id, worker)  # only the holder releases
        with self.lock:
            del self.claims[page_id]
            page = self.page(page_id)
            if report:
                page.setdefault("reports", []).append({"worker": worker, "text": report, "t": time.time()})
            if page.get("state") in (None, "drafted", "in_progress", "changes_requested"):
                page["state"] = "translated"
            self._save_meta()

    def review(self, page_id: str, worker: str, verdict: str, notes: str) -> dict:
        """approved, or changes_requested with notes the next worker on the page reads."""
        if verdict not in ("approved", "changes_requested"):
            raise StudioError("verdict is 'approved' or 'changes_requested'")
        if verdict == "changes_requested" and not notes.strip():
            raise StudioError("say what to change: notes")
        self._drafted(page_id)
        with self.lock:
            page = self.page(page_id)
            page.setdefault("reviews", []).append(
                {"worker": worker, "verdict": verdict, "notes": notes, "t": time.time()}
            )
            page["state"] = verdict
            self._save_meta()
            return {"page": page_id, "state": verdict}

    def notes(self) -> str:
        path = self.dir / "notes.md"
        return path.read_text() if path.exists() else ""

    def update_notes(self, text: str, mode: str) -> str:
        if mode not in ("replace", "append"):
            raise StudioError("mode is 'replace' or 'append'")
        with self.lock:
            new = text if mode == "replace" else (self.notes().rstrip() + "\n\n" + text).strip()
            (self.dir / "notes.md").write_text(new + "\n")
            return new

    # ------------------------------------------------------------------ drafting
    def start_draft(self, pages: list[str] | None, redo: bool) -> None:
        with self.lock:
            if self.job is not None and self.job.status in ("queued", "running"):
                return
            todo = [p for p in (pages or self.pages) if redo or self.page(p).get("draft") != "drafted"]
            for p in todo:
                self.meta["pages"][p].update(draft="queued", error=None)
            self._save_meta()
            if todo:
                self.job = self.engine.jobs.submit(
                    lambda cancel: self._draft_all(todo, cancel), Priority.BATCH, label=f"draft {self.id}"
                )

    def _draft_all(self, pages: list[str], cancel) -> None:
        for page_id in pages:
            if cancel.is_cancelled():
                with self.lock:
                    for p in pages:
                        if self.meta["pages"][p]["draft"] == "queued":
                            self.meta["pages"][p]["draft"] = "pending"
                    self._save_meta()
                return
            with self.lock:
                self.meta["pages"][page_id]["draft"] = "drafting"
                self._save_meta()
            t0 = time.time()
            try:
                with self._page_locks[page_id]:
                    regions = self._draft(page_id, cancel)
                self.log("studio", "draft", page_id, True, f"{regions} regions", (time.time() - t0) * 1000)
            except Exception as e:
                with self.lock:
                    self.meta["pages"][page_id].update(draft="error", error=f"{type(e).__name__}: {e}")
                    self._save_meta()
                self.log(
                    "studio", "draft", page_id, False, f"{type(e).__name__}: {e}", (time.time() - t0) * 1000
                )

    def _draft(self, page_id: str, cancel) -> int:
        """Detect, clean, read and size one page; record it; render it (no text yet)."""
        from mangatl.common.image_utils import convert_image_to_target_mode
        from mangatl.pipeline import page as page_pipeline
        from mangatl.pipeline import stages
        from mangatl.project import capture

        config = self.config.with_models(self.engine.models(Priority.BATCH))
        path = self.folder / page_id
        out = self.output_path(page_id)
        with Image.open(path) as opened:
            image_format = opened.format
            mime_type, cv2_ext = stages.get_image_encoding_params(image_format)
            target_mode = page_pipeline._resolve_target_mode(config, path, out)
            image = convert_image_to_target_mode(opened, target_mode, False)
        image, _ = page_pipeline._apply_pre_upscale_if_needed(image, config)
        w = stages.WorkPage(
            config=config,
            image=image,
            processing_scale=page_pipeline._resolve_processing_scale(image, config),
            target_mode=target_mode,
            mime_type=mime_type,
            cv2_ext=cv2_ext,
            image_format=image_format,
            source_path=path,
            context_image=image.copy(),
            cancel=cancel,
        )
        stages.detect(w)
        if w.bubbles or w.outside:
            stages.clean(w)
            stages.prepare(w, refine_masks=True)
            if w.ordered:
                stages.ocr(w)
                w.translations = [""] * len(w.ordered)
                stages.typeset(w)
        capture.capture(self.store, page_id, work=w, source_path=path, output_path=out)
        outcomes: dict = {}
        rendered = stages.render(w, outcomes=outcomes) if w.typeset_done else stages.result(w)
        self._write(out, rendered, w.target_mode)
        with self._work_lock:
            self._work.pop(page_id, None)
        with self.lock:
            self.meta["pages"][page_id].update(
                draft="drafted",
                error=None,
                outcomes=outcomes,
                version=self.page(page_id).get("version", 0) + 1,
                state="drafted",
            )
            self._save_meta()
        self._summarize(page_id)
        return len(w.ordered)

    def _write(self, out: Path, image: Image.Image, mode: str) -> None:
        from mangatl.common.image_utils import save_image_with_compression

        out.parent.mkdir(parents=True, exist_ok=True)
        save_image_with_compression(
            image if image.mode == mode else image.convert(mode),
            out,
            jpeg_quality=self.config.output.jpeg_quality,
            png_compression=self.config.output.png_compression,
            verbose=False,
        )

    def wait_draft(self, seconds: float) -> None:
        job = self.job
        if job is None or seconds <= 0:
            return
        try:
            job.result(timeout=seconds)
        except Exception:  # timeout, or a failed job: the pages say which
            pass

    # ------------------------------------------------------------------ editing
    def work(self, page_id: str):
        """The stored page, loaded for the stages (cached)."""
        from mangatl.pipeline import stored

        with self._work_lock:
            w = self._work.get(page_id)
            if w is not None:
                self._work.move_to_end(page_id)
                return w
        # the studio only edits text and lettering after drafting: no OCR / LLM crops needed
        w = stored.load(self.store, page_id, self.engine.models(Priority.INTERACTIVE), crops=False)
        # the project's style sheet is the one every page follows (pages keep the settings they
        # were drafted with otherwise)
        ts = self.config.typesetting
        drafted_base = w.config.typesetting.base_size
        w.config = w.config.updated(
            **{
                "runtime.verbose": False,
                "typesetting.presets": ts.presets,
                "typesetting.base_size": ts.base_size,
                "typesetting.synthetic_styles": ts.synthetic_styles,
                "typesetting.em_dash": ts.em_dash,
            }
        )
        if w.typeset_done and drafted_base != ts.base_size:
            from mangatl.pipeline import stages

            stages.typeset(w)
        with self._work_lock:
            self._work[page_id] = w
            while len(self._work) > WORK_CACHE:
                self._work.popitem(last=False)
        return w

    def set_translations(
        self,
        page_id: str,
        translations: dict[str, str],
        worker: str,
        styles: dict | None = None,
        speakers: dict[str, str] | None = None,
    ) -> dict:
        """Set translations (and styles, as set_style; speakers, as annotate) and re-render."""
        self._drafted(page_id)
        self._may_edit(page_id, worker)
        with self._page_locks[page_id]:
            w = self.work(page_id)
            ids = [item["region_id"] for item in w.ordered]
            unknown = [r for r in [*translations, *(styles or {}), *(speakers or {})] if r not in ids]
            if unknown:
                raise StudioError(f"no translatable region {unknown} on {page_id}; its regions: {ids}")
            for rid, name in (speakers or {}).items():
                _set_field(w.ordered[ids.index(rid)], "speaker", name)
            new_styles = {
                rid: self._merged_style(w.ordered[ids.index(rid)], st) for rid, st in (styles or {}).items()
            }
            if len(w.translations) != len(ids):
                w.translations = [""] * len(ids)
            for rid, text in translations.items():
                w.translations[ids.index(rid)] = text.strip()
            for rid, st in new_styles.items():
                _set_style(w.ordered[ids.index(rid)], st)
            changed: list[str] = list(dict.fromkeys([*translations, *new_styles, *(speakers or {})]))
            ran = {"translate", "typeset"} if translations else {"typeset"}
            self._commit(
                page_id,
                w,
                worker,
                f"{'translations' if translations else 'styles'} by {worker}",
                ran,
                changed,
            )
        return self.page_info(page_id, only=changed)

    def set_style(self, page_id: str, styles: dict, worker: str) -> dict:
        return self.set_translations(page_id, {}, worker, styles)

    def _merged_style(self, item: dict, change: dict | None) -> dict | None:
        """A region's own style with `change` merged in (a field set to None is cleared; None for
        the whole change clears the style); validated against the fonts and the sheet."""
        if change is None:
            return None
        style = {**(item.get("style") or {}), **change}
        style = {k: v for k, v in style.items() if v is not None}
        try:
            checked = LetterStyle.model_validate(style)
        except ValueError as e:
            raise StudioError(f"bad style for {item['region_id']}: {e}") from e
        self._check_style(checked, item["region_id"])
        return checked.model_dump(exclude_none=True) or None

    def _check_style(self, style: LetterStyle, where: str) -> None:
        if style.font and style.font not in font_packs(FONTS_DIR):
            raise StudioError(f"{where}: no font {style.font!r}; fonts: {sorted(font_packs(FONTS_DIR))}")
        presets = self.config.typesetting.presets
        if style.preset and style.preset not in presets:
            raise StudioError(f"{where}: no preset {style.preset!r}; the sheet has {sorted(presets)}")

    def _commit(
        self, page_id: str, w, worker: str, label: str, ran: set[str], regions: list[str] | None
    ) -> None:
        """Render the working page, record its changes, write the output (the page lock is held)."""
        from mangatl.pipeline import stages, stored

        if not w.typeset_done:
            stages.typeset(w)
        outcomes: dict = {}
        image = stages.render(w, outcomes=outcomes)
        stored.save(self.store, page_id, w, author="agent", label=label, ran=ran, region_ids=regions)
        self._write(self.output_path(page_id), image, w.target_mode)
        with self.lock:
            page = self.page(page_id)
            page.update(outcomes=outcomes, version=page.get("version", 0) + 1, rendered=time.time())
            self._save_meta()
        self._summarize(page_id)

    def _summarize(self, page_id: str) -> None:
        """The page's counts as get_page computes them, for project_status (one source of truth)."""
        info = self.page_info(page_id)
        with self.lock:
            self.page(page_id)["summary"] = {
                k: info[k] if k != "problems" else len(info[k])
                for k in ("regions_total", "translated", "dismissed", "problems")
            }
            self._save_meta()

    def annotate(self, page_id: str, regions: dict[str, dict], worker: str) -> dict:
        """Per region, any of: speaker, source (the corrected Japanese), dismissed (why it isn't
        text; "" or None restores it), link (a name shared by the bubbles of one sentence; None
        unlinks). Re-renders the page."""
        self._drafted(page_id)
        self._may_edit(page_id, worker)
        allowed = {"speaker", "source", "dismissed", "link"}
        with self._page_locks[page_id]:
            w = self.work(page_id)
            ids = [item["region_id"] for item in w.ordered]
            for rid, fields in regions.items():
                if rid not in ids:
                    raise StudioError(f"no translatable region {rid} on {page_id}; its regions: {ids}")
                extra = set(fields) - allowed
                if extra:
                    raise StudioError(f"{rid}: unknown fields {sorted(extra)}; allowed: {sorted(allowed)}")
            if len(w.ocr_texts) != len(ids):
                w.ocr_texts = [item.get("ocr_text") or "" for item in w.ordered]
            ran: set[str] = {"typeset"}
            for rid, fields in regions.items():
                item = w.ordered[ids.index(rid)]
                for key in ("speaker", "dismissed", "link"):
                    if key in fields:
                        _set_field(item, key, fields[key])
                if "source" in fields:
                    text = (fields["source"] or "").strip()
                    w.ocr_texts[ids.index(rid)] = text
                    item["ocr_text"] = text
                    ran.add("ocr")
            self._commit(page_id, w, worker, f"regions annotated by {worker}", ran, list(regions))
        return self.page_info(page_id, only=list(regions))

    def set_order(self, page_id: str, order: list[str], worker: str) -> dict:
        """Put the page's regions in this reading order (every translatable region, once)."""
        self._drafted(page_id)
        self._may_edit(page_id, worker)
        with self._page_locks[page_id]:
            w = self.work(page_id)
            ids = [item["region_id"] for item in w.ordered]
            if sorted(order) != sorted(ids):
                missing, extra = set(ids) - set(order), set(order) - set(ids)
                raise StudioError(
                    f"give every region once; missing {sorted(missing)}, unknown {sorted(extra)}"
                )
            n = len(ids)
            if len(w.translations) != n:
                w.translations = [""] * n
            if len(w.ocr_texts) != n:
                w.ocr_texts = [item.get("ocr_text") or "" for item in w.ordered]
            pick = [ids.index(r) for r in order]
            w.ordered = [w.ordered[i] for i in pick]
            w.translations = [w.translations[i] for i in pick]
            w.ocr_texts = [w.ocr_texts[i] for i in pick]
            self._commit(page_id, w, worker, f"reading order by {worker}", {"typeset"}, None)
        return self.page_info(page_id)

    def history(self, page_id: str, region_id: str) -> list[dict]:
        """A region's past translations, source reads, speakers, styles and dismissals."""
        self._drafted(page_id)
        keep = {"translation", "ocr", "speaker", "style", "dismissed", "link"}
        return [
            {"t": round(h["t"]), "by": h["label"], "field": h["field"], "old": h["old"], "new": h["new"]}
            for h in self.store.region_history(page_id, region_id)
            if h["field"] in keep
        ]

    def preview(self, page_id: str, region_id: str, variants: list[dict]) -> tuple[Image.Image, list[dict]]:
        """Render up to MAX_PREVIEWS variants ({"text", "style"}: the style merges into the region's
        own) of one region without keeping any; returns them side by side, and their outcomes."""
        from mangatl.pipeline import stages

        self._drafted(page_id)
        if not variants:
            raise StudioError("give at least one variant ({text, style})")
        variants = variants[:MAX_PREVIEWS]
        with self._page_locks[page_id]:
            w = self.work(page_id)
            ids = [item["region_id"] for item in w.ordered]
            if region_id not in ids:
                raise StudioError(f"no translatable region {region_id} on {page_id}; its regions: {ids}")
            idx = ids.index(region_id)
            item = w.ordered[idx]
            if len(w.translations) != len(ids):
                w.translations = [""] * len(ids)
            if not w.typeset_done:
                stages.typeset(w)
            text0, style0 = w.translations[idx], item.get("style")
            crops, results = [], []
            try:
                for v in variants:
                    w.translations[idx] = str(v.get("text", text0)).strip()
                    _set_style(
                        item, self._merged_style(item, v.get("style") or {}) if "style" in v else style0
                    )
                    outcomes: dict = {}
                    image = stages.render(w, outcomes=outcomes)
                    o = outcomes.get(region_id, {})
                    results.append(
                        {
                            "variant": len(results) + 1,
                            **o,
                            "problem": _problem({"translation": w.translations[idx], **o}),
                        }
                    )
                    crops.append(
                        _crop_around(
                            image, item["bbox"], o.get("text_box"), (item.get("style") or {}).get("box")
                        )
                    )
            finally:
                w.translations[idx] = text0
                _set_style(item, style0)
        return _side_by_side(crops, [str(r["variant"]) for r in results]), results

    # ------------------------------------------------------------------ style sheet
    def _with_sheet(self, config: Settings) -> Settings:
        sheet = self.meta.get("style_sheet") or {}
        changes: dict[str, Any] = {}
        if "presets" in sheet:
            changes["typesetting.presets"] = {
                k: LetterStyle.model_validate(v) for k, v in sheet["presets"].items()
            }
        if "base_size" in sheet:
            changes["typesetting.base_size"] = sheet["base_size"]
        return config.updated(**changes) if changes else config

    def style_sheet(self) -> dict:
        ts = self.config.typesetting
        return {
            "base_size": ts.base_size,
            "presets": {k: v.model_dump(exclude_none=True) for k, v in ts.presets.items()},
            "fonts": font_packs(FONTS_DIR),
            "note": "bubbles use the 'dialogue' preset and text on the art 'outside' (if the sheet has it) "
            "unless a region names another; base_size None: each page's dialogue size from its lettering",
        }

    def set_style_sheet(
        self, presets: dict | None, base_size: float | None, replace: bool, worker: str
    ) -> dict:
        """Change the chapter's presets (merged, a preset set to None is removed; replace: these
        only) and base size (0: back to each page's own), then re-render every drafted page."""
        from mangatl.pipeline import stages

        ts = self.config.typesetting
        current = {} if replace else {k: v.model_dump(exclude_none=True) for k, v in ts.presets.items()}
        for name, value in (presets or {}).items():
            if value is None:
                current.pop(name, None)
                continue
            try:
                style = LetterStyle.model_validate(value)
            except ValueError as e:
                raise StudioError(f"bad preset {name!r}: {e}") from e
            if style.preset or style.box or style.offset:
                raise StudioError(f"preset {name!r}: presets can't name a preset or place text (box, offset)")
            if style.font and style.font not in font_packs(FONTS_DIR):
                raise StudioError(
                    f"preset {name!r}: no font {style.font!r}; fonts: {sorted(font_packs(FONTS_DIR))}"
                )
            current[name] = style.model_dump(exclude_none=True)
        base = ts.base_size if base_size is None else (base_size or None)
        with self.lock:
            self.meta["style_sheet"] = {"presets": current, "base_size": base}
            self._save_meta()
            self.config = self._with_sheet(self.config)
        typeset_again = base != ts.base_size
        changes = {
            "typesetting.presets": {k: LetterStyle.model_validate(v) for k, v in current.items()},
            "typesetting.base_size": base,
        }
        pages = {}
        for page_id in self.pages:
            if self.page(page_id).get("draft") != "drafted":
                continue
            with self._page_locks[page_id]:
                w = self.work(page_id)
                w.config = w.config.updated(**changes)
                if typeset_again and w.typeset_done:
                    stages.typeset(w)
                self._commit(page_id, w, worker, f"style sheet by {worker}", {"typeset"}, None)
            pages[page_id] = len(self.page_info(page_id)["problems"])
        return {**self.style_sheet(), "problems_per_page": pages}

    # ------------------------------------------------------------------ reading
    def page_info(self, page_id: str, only: list[str] | None = None) -> dict:
        """The page as an agent reads it: regions in reading order with their source text,
        translation and how the translation came out."""
        page = self._drafted(page_id)
        with self._page_locks[page_id]:
            w = self.work(page_id)
            ordered = list(w.ordered)
            translations = (
                list(w.translations) if len(w.translations) == len(ordered) else [""] * len(ordered)
            )
            ocr = list(w.ocr_texts) if len(w.ocr_texts) == len(ordered) else [""] * len(ordered)
            size = w.image.size
            dialogue = getattr(w.sizes, "main_max_font", None)
            presets = w.config.typesetting.presets
            glyphs = dict(w.glyphs or {})
        outcomes = page.get("outcomes") or {}
        links: dict[str, list[str]] = defaultdict(list)
        for item in ordered:
            if item.get("link"):
                links[item["link"]].append(item["region_id"])
        regions, problems = [], []
        for n, (item, text, read) in enumerate(zip(ordered, translations, ocr, strict=True), 1):
            rid = item["region_id"]
            o = outcomes.get(rid, {})
            region = {
                "id": rid,
                "order": n,
                "kind": "outside" if item.get("is_outside_text") else "bubble",
                "bbox": [int(v) for v in item["bbox"]],
                "ocr": read,
                "translation": text,
                "outcome": "dismissed" if item.get("dismissed") else o.get("outcome", "empty"),
                "font_size": o.get("font_size"),
                "max_font": o.get("max_font"),
            }
            if item.get("panel_id") is not None:
                region["panel"] = int(item["panel_id"])
            if item.get("class"):
                region["detected_as"] = item["class"]
            source_size = _source_size(glyphs, item["bbox"])
            if source_size:
                region["source_size"] = source_size
            for key in ("speaker", "dismissed"):
                if item.get(key):
                    region[key] = item[key]
            if item.get("link"):
                region["linked_with"] = [r for r in links[item["link"]] if r != rid]
            for key in ("fill", "margin", "lines"):
                if o.get(key) is not None:
                    region[key] = o[key]
            own = item.get("style")
            style = resolve(own, presets, region["kind"])
            if style is not None:
                region["preset"] = style.preset or ("outside" if region["kind"] == "outside" else "dialogue")
            if own:
                region["style"] = own
            problem = None if item.get("dismissed") else _problem(region)
            if problem:
                region["problem"] = problem
                problems.append(f"{rid}: {problem}")
            elif not item.get("dismissed"):
                hint = _hint(region)
                if hint:
                    region["hint"] = hint
            if only is None or rid in only:
                regions.append(region)
        dismissed = sum(1 for item in ordered if item.get("dismissed"))
        info = {
            "project": self.id,
            "page": page_id,
            "state": page.get("state", "drafted"),
            "size": list(size),
            "dialogue_font_size": dialogue,
            "claimed_by": self.claimed_by(page_id),
            "translated": sum(
                1 for t, item in zip(translations, ordered, strict=True) if t and not item.get("dismissed")
            ),
            "regions_total": len(ordered) - dismissed,
            "dismissed": dismissed,
            "problems": problems,
            "regions": regions,
            "output": str(self.output_path(page_id)),
        }
        if page.get("reviews"):
            info["reviews"] = page["reviews"]
        if page.get("reports"):
            info["reports"] = page["reports"]
        if only is not None:
            info["note"] = "only the regions just changed are listed; get_page shows all"
        return info

    def status(self) -> dict:
        pages = []
        for pid in self.pages:
            page = self.meta["pages"][pid]
            outcomes = page.get("outcomes") or {}
            entry: dict[str, Any] = {"page": pid, "draft": page.get("draft")}
            if page.get("error"):
                entry["error"] = page["error"]
            if page.get("draft") == "drafted":
                entry["state"] = page.get("state", "drafted")
                summary = page.get("summary") or {
                    "regions_total": len(outcomes),
                    "translated": sum(1 for o in outcomes.values() if o.get("outcome") != "empty"),
                    "problems": sum(1 for o in outcomes.values() if o.get("outcome") in PROBLEMS),
                }
                entry.update(summary)
            holder = self.claimed_by(pid)
            if holder:
                entry["claimed_by"] = holder
            if page.get("reports"):
                entry["reports"] = [r["worker"] for r in page["reports"]]
            pages.append(entry)
        job = self.job.info() if self.job is not None else None
        return {
            "project": self.id,
            "folder": str(self.folder),
            "output": str(self.out),
            "draft_job": job,
            "pages": pages,
            "notes_chars": len(self.notes()),
        }

    # ------------------------------------------------------------------ the chapter as text
    def script(self, pages: list[str] | None = None, speaker: str | None = None) -> str:
        """The chapter as a script: per page, each region in reading order on one line:
        number, id, [speaker], source → translation, and what's wrong with it."""
        lines = []
        for page_id in pages or self.pages:
            if self.page(page_id).get("draft") != "drafted":
                if speaker is None:
                    lines.append(f"## {page_id} (not drafted)")
                continue
            info = self.page_info(page_id)
            rows = [
                r
                for r in info["regions"]
                if speaker is None or (r.get("speaker") or "").lower() == speaker.lower()
            ]
            if speaker is not None and not rows:
                continue
            lines.append(
                f"## {page_id} ({info['state']}, {info['translated']}/{info['regions_total']} translated)"
            )
            for r in rows:
                who = f"[{r['speaker']}] " if r.get("speaker") else ""
                kind = " (art)" if r["kind"] == "outside" else ""
                link = f" (+{','.join(r['linked_with'])})" if r.get("linked_with") else ""
                if r.get("dismissed"):
                    lines.append(f"{r['order']}. {r['id']}{kind}: dismissed ({r['dismissed']})")
                    continue
                tail = f"   !! {r['problem']}" if r.get("problem") else ""
                lines.append(
                    f"{r['order']}. {r['id']}{kind}{link} {who}{r['ocr']} → {r['translation'] or '—'}{tail}"
                )
        return "\n".join(lines)

    def context(self, page_id: str) -> str:
        """The pages either side of this one, as script (their latest translations, or the source)."""
        i = self.pages.index(page_id) if page_id in self.pages else -1
        if i < 0:
            self.page(page_id)  # raises
        around = [
            p
            for p in (
                self.pages[i - 1] if i > 0 else None,
                self.pages[i + 1] if i + 1 < len(self.pages) else None,
            )
            if p
        ]
        return self.script(around) if around else "(no other pages)"

    # ------------------------------------------------------------------ glossary
    def glossary(self) -> list[dict]:
        return list(self.meta.get("glossary") or [])

    def set_glossary(self, entries: list[dict], replace: bool) -> list[dict]:
        """Entries {source, english, kind?, note?} by source term; english None removes one."""
        current = {} if replace else {e["source"]: e for e in self.glossary()}
        for e in entries:
            if not isinstance(e, dict) or not e.get("source"):
                raise StudioError("each entry needs 'source' (the Japanese) and 'english'")
            if e.get("english") in (None, ""):
                current.pop(e["source"], None)
                continue
            extra = set(e) - {"source", "english", "kind", "note"}
            if extra:
                raise StudioError(f"unknown entry fields {sorted(extra)}")
            current[e["source"]] = {k: e[k] for k in ("source", "english", "kind", "note") if e.get(k)}
        with self.lock:
            self.meta["glossary"] = list(current.values())
            self._save_meta()
        return self.glossary()

    def check_consistency(self) -> dict:
        """What a consistency pass looks for: glossary terms whose chosen English is missing where
        the source has them, variants of a chosen English (Tortiyanos vs Tortiyano), honorifics
        left in, lines without a speaker, and the open problems."""
        import re

        regions = []
        for page_id in self.pages:
            if self.page(page_id).get("draft") == "drafted":
                regions += [
                    (page_id, r) for r in self.page_info(page_id)["regions"] if not r.get("dismissed")
                ]
        missing, variants, honorifics, problems = [], defaultdict(set), [], []
        no_speaker = [f"{p} {r['id']}" for p, r in regions if r["kind"] == "bubble" and not r.get("speaker")]
        for e in self.glossary():
            english = e["english"]
            word = re.compile(rf"\b{re.escape(english)}\b", re.I)
            variant = re.compile(rf"\b{re.escape(english)}(\w+)\b", re.I)
            for p, r in regions:
                if e["source"] in (r["ocr"] or "") and r["translation"] and not word.search(r["translation"]):
                    missing.append(
                        f"{p} {r['id']}: source has {e['source']} but not '{english}': {r['translation']}"
                    )
                for m in variant.finditer(r["translation"] or ""):
                    variants[english].add(f"{m.group(0)} ({p} {r['id']})")
        hon = re.compile(r"\b\w+-(san|kun|chan|sama|senpai|sensei|dono|tan)\b", re.I)
        for p, r in regions:
            for m in hon.finditer(r["translation"] or ""):
                honorifics.append(f"{p} {r['id']}: {m.group(0)}")
            if r.get("problem"):
                problems.append(f"{p} {r['id']}: {r['problem']}")
        speakers: dict[str, int] = defaultdict(int)
        for _, r in regions:
            if r.get("speaker"):
                speakers[r["speaker"]] += 1
        return {
            "glossary_missing": missing,
            "variants": {k: sorted(v) for k, v in variants.items()},
            "honorifics": honorifics,
            "speakers": dict(sorted(speakers.items(), key=lambda kv: -kv[1])),
            "bubbles_without_speaker": no_speaker,
            "problems": problems,
        }

    # ------------------------------------------------------------------ finishing
    def export(self, cbz: bool) -> dict:
        """The script (script.md beside the pages) and optionally a .cbz of the rendered pages,
        and what is still unresolved."""
        import zipfile

        script = self.dir / "script.md"
        script.write_text(f"# {self.folder.name}\n\n{self.script()}\n")
        out: dict[str, Any] = {"pages": str(self.out), "script": str(script)}
        if cbz:
            archive = self.dir / f"{self.folder.name}.cbz"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
                for n, page_id in enumerate(self.pages, 1):
                    path = self.output_path(page_id)
                    if path.exists():
                        z.write(path, f"{n:04d}_{Path(page_id).stem}.png")
            out["cbz"] = str(archive)
        unresolved = []
        for entry in self.status()["pages"]:
            if entry["draft"] != "drafted":
                unresolved.append(f"{entry['page']}: not drafted ({entry['draft']})")
                continue
            if entry.get("problems"):
                unresolved.append(f"{entry['page']}: {entry['problems']} problems")
            if entry.get("state") != "approved":
                unresolved.append(f"{entry['page']}: {entry.get('state')}, not approved")
            if entry.get("claimed_by"):
                unresolved.append(f"{entry['page']}: still claimed by {entry['claimed_by']}")
        out["unresolved"] = unresolved
        return out

    def image(self, page_id: str, layer: str) -> Image.Image:
        """original: the source; cleaned: text removed, nothing drawn; rendered: the latest render."""
        if layer == "original":
            with Image.open(self.folder / page_id) as im:
                return im.convert("RGB")
        self._drafted(page_id)
        if layer == "rendered" and self.output_path(page_id).exists():
            with Image.open(self.output_path(page_id)) as im:
                return im.convert("RGB")
        if layer not in ("cleaned", "rendered"):
            raise StudioError("layer is 'original', 'cleaned' or 'rendered'")
        from mangatl.clean.outside_text import paint_pending_osb_fills

        with self._page_locks[page_id]:
            w = self.work(page_id)
            if w.canvas is None:
                return w.image.convert("RGB")
            return paint_pending_osb_fills(w.canvas.copy(), [dict(i) for i in w.outside]).convert("RGB")

    def regions_for_overlay(self, page_id: str) -> tuple[tuple[int, int], list[dict]]:
        info = self.page_info(page_id)
        return tuple(info["size"]), info["regions"]  # type: ignore[return-value]


def _set_field(item: dict, key: str, value) -> None:
    if value in (None, ""):
        item.pop(key, None)
    else:
        item[key] = str(value).strip()


def _set_style(item: dict, style: dict | None) -> None:
    if style:
        item["style"] = style
    else:
        item.pop("style", None)


def _source_size(glyphs: dict, bbox) -> float | None:
    """The source lettering's size in this region relative to the page's typical size."""
    from mangatl.typeset.page import _lookup_bubble_glyph

    page, own = glyphs.get("page"), glyphs.get("own")
    if not page or not own:
        return None
    est = _lookup_bubble_glyph(own, bbox)
    return round(est[0] / page, 2) if est and est[1] >= 3 else None


def _hint(region: dict) -> str | None:
    """Not a problem, but worth a look: short text left small in a big bubble."""
    fill, size, cap = region.get("fill"), region.get("font_size"), region.get("max_font")
    if (
        region["outcome"] == "fit"
        and fill is not None
        and size
        and cap
        and fill < UNDERFILLED
        and size >= cap - 0.5
    ):
        return f"the text covers {fill:.0%} of the bubble at its size cap; a bigger scale may suit it"
    return None


def _crop_around(image: Image.Image, bbox, text_box=None, box=None, pad: int = 16) -> Image.Image:
    xs, ys = [bbox[0], bbox[2]], [bbox[1], bbox[3]]
    for b in (text_box, box):
        if b:
            xs += [b[0], b[2]]
            ys += [b[1], b[3]]
    return image.crop(
        (
            max(0, min(xs) - pad),
            max(0, min(ys) - pad),
            min(image.width, max(xs) + pad),
            min(image.height, max(ys) + pad),
        )
    ).convert("RGB")


def _side_by_side(images: list[Image.Image], labels: list[str], max_side: int = 900) -> Image.Image:
    gap, label_h = 10, 22
    h = max(im.height for im in images)
    out = Image.new(
        "RGB", (sum(im.width for im in images) + gap * (len(images) - 1), h + label_h), (128, 128, 128)
    )
    draw = ImageDraw.Draw(out)
    x = 0
    for im, label in zip(images, labels, strict=True):
        out.paste(im, (x, label_h))
        draw.text((x + 4, 2), label, fill=(255, 255, 255), font=_font(16))
        x += im.width + gap
    scale = min(max_side / max(out.size), 3.0)  # small regions are enlarged, up to 3x
    if scale != 1:
        out = out.resize((round(out.width * scale), round(out.height * scale)), Image.Resampling.LANCZOS)
    return out


def font_samples(max_width: int = 900) -> Image.Image:
    """One line of sample lettering per font pack, with the styles it has."""
    from PIL import ImageFont as F

    from mangatl.typeset.fonts import find_font_variants

    rows = []
    for name, styles in font_packs(FONTS_DIR).items():
        regular = find_font_variants(str(FONTS_DIR / name), verbose=False)["regular"]
        font = F.truetype(str(regular), 34)
        row = Image.new("RGB", (max_width, 76), "white")
        d = ImageDraw.Draw(row)
        d.text((10, 4), f"{name}  ({', '.join(styles)})", fill=(110, 110, 110), font=_font(14))
        d.text((10, 26), "WHAT?! I'm not going... Hey, wait!", fill=(0, 0, 0), font=font)
        rows.append(row)
    out = Image.new("RGB", (max_width, sum(r.height for r in rows)), "white")
    for k, r in enumerate(rows):
        out.paste(r, (0, k * 76))
    return out


def _problem(region: dict) -> str | None:
    if not region["translation"]:
        return "not translated"
    if region["outcome"] in PROBLEMS:
        return PROBLEMS[region["outcome"]]
    size, cap = region.get("font_size"), region.get("max_font")
    if size and cap and size < SMALL_TEXT * cap:
        return f"small text: {size:.0f}px, {size / cap:.0%} of this region's size cap ({cap:.0f}px); shorten or rebreak"
    return None


# --------------------------------------------------------------------------- images
OUTCOME_COLORS = {
    "fit": (46, 160, 67),
    "below_min_font": (230, 140, 20),
    "kept_original_no_fit": (220, 40, 40),
    "failed": (220, 40, 40),
    "kept_original": (140, 90, 200),
    "empty": (90, 140, 230),
}


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def overlay(image: Image.Image, work_size: tuple[int, int], regions: list[dict]) -> Image.Image:
    """Region boxes with their reading order and id, coloured by outcome."""
    image = image.convert("RGB")
    sx, sy = image.width / work_size[0], image.height / work_size[1]
    draw = ImageDraw.Draw(image)
    width = max(2, round(image.width / 400))
    font = _font(max(14, round(image.width / 55)))
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        color = (
            OUTCOME_COLORS.get(r["outcome"], (200, 200, 200)) if r["translation"] else OUTCOME_COLORS["empty"]
        )
        draw.rectangle(box, outline=color, width=width)
        label = f"{r['order']} {r['id']}"
        tb = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle((tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2), fill=color)
        draw.text((box[0], box[1]), label, fill=(255, 255, 255), font=font)
    return image


def fit(image: Image.Image, max_side: int) -> Image.Image:
    scale = max_side / max(image.size)
    if scale >= 1:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def encode(image: Image.Image, fmt: str = "JPEG", quality: int = 88) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def region_view(
    project: StudioProject,
    page_id: str,
    region_id: str,
    layers: tuple[str, ...] = ("original", "rendered"),
    zoom: float = 1.0,
    boxes: bool = False,
) -> tuple[Image.Image, dict]:
    """The region in each layer side by side, with some context around it; `boxes` draws the
    region's box (blue) and where its text landed (green)."""
    info = project.page_info(page_id)
    region = next((r for r in info["regions"] if r["id"] == region_id), None)
    if region is None:
        raise StudioError(
            f"no region {region_id} on {page_id}; regions: {[r['id'] for r in info['regions']]}"
        )
    bad = [layer for layer in layers if layer not in ("original", "cleaned", "rendered")]
    if bad or not layers:
        raise StudioError("layers are 'original', 'cleaned', 'rendered'")
    work_w, work_h = info["size"]
    x1, y1, x2, y2 = region["bbox"]
    pad = max(16, round(0.15 * max(x2 - x1, y2 - y1)))
    text_box = (project.page(page_id).get("outcomes") or {}).get(region_id, {}).get("text_box")
    crops = []
    for layer in layers:
        im = project.image(page_id, layer)
        sx, sy = im.width / work_w, im.height / work_h
        box = (
            max(0, round((x1 - pad) * sx)),
            max(0, round((y1 - pad) * sy)),
            min(im.width, round((x2 + pad) * sx)),
            min(im.height, round((y2 + pad) * sy)),
        )
        crop = im.crop(box)
        if boxes:
            d = ImageDraw.Draw(crop)

            def rect(b, color):
                d.rectangle(
                    (b[0] * sx - box[0], b[1] * sy - box[1], b[2] * sx - box[0], b[3] * sy - box[1]),
                    outline=color,
                    width=1,
                )

            rect(region["bbox"], (60, 120, 255))
            if text_box and layer == "rendered":
                rect(text_box, (40, 180, 70))
        crops.append(crop)
    image = _side_by_side(crops, list(layers), max_side=round(640 * max(0.5, min(zoom, 3.0))))
    return image, region
