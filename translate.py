#!/usr/bin/env python3
"""Fully local batch manga translator.

Pipeline (all on this machine):
  detection   YOLO speech-bubble seg + RT-DETR conjoined bubbles + panel YOLO (reading order)
  OCR         Hayai OCR v2.5 Nova (+ manga-ocr cross-check) for Japanese, PaddleOCR-VL 1.6 otherwise
  cleaning    flat-fill for bubbles, SAM-TS-L glyph masks + LaMa-manga inpainting for the rest
  translate   Gemma 4 26B-A4B (QAT) served by a local llama.cpp server
  typeset     auto font size, hyphenation, emphasis, sound-effect labels

The pipeline lives in ./mangatl (derived from MangaTranslator, see mangatl/NOTICE). This wrapper
turns its options into one Settings object (mangatl/presets.py), writes it to
<output>/settings.json, starts/stops llama-server and runs the pipeline per chapter.

Usage:
  uv run python translate.py --input chapter01/ --output chapter01_en/
  uv run python translate.py --input vol1.cbz --output vol1_en/ --target English
  uv run python translate.py ... --set typesetting.bubbles.max_font_size=28   # any setting
  uv run python translate.py ... --show-settings                              # see them all
"""

from __future__ import annotations

import argparse
import atexit
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mangatl import presets
from mangatl.settings import FONTS_DIR, Settings

MODELS_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
LLAMA_SERVER = ROOT / "bin" / "llama.cpp" / "llama-server"

OSB_TEXT_MODEL = MODELS_DIR / "detection" / "yolo" / "animetext_yolov12x.pt"
OSB_TEXT_URL = "https://huggingface.co/deepghs/AnimeText_yolo/resolve/main/yolo12x_animetext/model.pt"


def log(msg: str) -> None:
    print(f"[translate] {msg}", flush=True)


# --------------------------------------------------------------------------- llama-server


def server_healthy(base_url: str) -> bool:
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/health", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def start_llama_server(server, vision: bool, repo: str) -> subprocess.Popen:
    if not LLAMA_SERVER.exists():
        sys.exit(f"llama-server not found at {LLAMA_SERVER}")
    if not Path(server.gguf).exists():
        sys.exit(f"LLM weights not found: {server.gguf}")

    cmd = [
        str(LLAMA_SERVER),
        "-m",
        str(server.gguf),
        "--alias",
        presets.model_alias(server.gguf),
        "--host",
        "127.0.0.1",
        "--port",
        str(server.port),
        "-c",
        str(server.ctx),
        "-np",
        "1",
        "-ngl",
        "99",
        "-fa",
        "on",
        "--reasoning",
        "off",  # thinking burns tokens without improving translations
        "--no-webui",
    ]
    if vision:
        if not Path(server.mmproj).exists():
            sys.exit(
                f"Vision projector not found: {server.mmproj}\n"
                f"Download mmproj-F16.gguf from https://huggingface.co/{repo} to that path."
            )
        # Gemma 4 vision needs the whole image in one ubatch
        cmd += ["--mmproj", str(server.mmproj), "-b", "2048", "-ub", "2048"]
    else:
        cmd += ["--no-mmproj"]
    if server.cpu_moe:  # MoE models: keep expert weights of the first N layers in system RAM
        cmd += ["--n-cpu-moe", str(server.cpu_moe)]
    extra = shlex.split(server.extra_args)
    if "--fit" in extra or "-fit" in extra:  # let llama.cpp choose the GPU layer count
        i = cmd.index("-ngl")
        del cmd[i : i + 2]
    cmd += extra

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / "llama-server.log"
    log(f"Starting llama-server ({'vision' if vision else 'text-only'}), log: {log_path}")
    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)

    def _stop() -> None:
        if proc.poll() is None:
            log("Stopping llama-server")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        logf.close()

    atexit.register(_stop)

    url = f"http://127.0.0.1:{server.port}/v1"
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited early (code {proc.returncode}); see {log_path}")
        if server_healthy(url):
            log("llama-server ready")
            return proc
        time.sleep(1)
    sys.exit(f"llama-server did not become healthy in time; see {log_path}")


# --------------------------------------------------------------------------- helpers


def osb_model_available(offline: bool) -> bool:
    """AnimeText_yolo is gated on Hugging Face; only use OSB features if we can get it."""
    if OSB_TEXT_MODEL.exists():
        return True
    if offline:
        return False
    token = os.environ.get("HF_TOKEN")
    token_file = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    if not token and token_file.exists():
        token = token_file.read_text().strip()
    if not token:
        return False
    req = urllib.request.Request(OSB_TEXT_URL, method="HEAD", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def check_font_pack(name: str) -> None:
    p = Path(name)
    if not p.is_absolute():
        p = FONTS_DIR / name
    if not p.is_dir() or not (list(p.glob("*.ttf")) + list(p.glob("*.otf"))):
        packs = ", ".join(sorted(d.name for d in FONTS_DIR.iterdir() if d.is_dir()))
        sys.exit(f"Font pack '{name}' not found (folder of .ttf/.otf). Available: {packs}")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def image_dirs(root: Path) -> list[Path]:
    """root and every subfolder that directly contains images, in natural (chapter) order."""
    dirs = [
        d
        for d in [root, *root.rglob("*")]
        if d.is_dir() and any(f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in d.iterdir())
    ]
    return sorted(dirs, key=lambda d: natural_key(d.relative_to(root)))


def translated_name(img: Path, fmt: str) -> str:
    ext = img.suffix if fmt == "auto" else (".jpg" if fmt == "jpeg" else ".png")
    return f"{img.stem}_translated{ext}"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Fully local batch manga translator (mangatl pipeline + llama.cpp).",
        epilog="Every other setting: --set section.field=value (see --show-settings).",
    )
    p.add_argument("--input", "-i", help="Folder of images, a single image, or a .zip/.cbz archive")
    p.add_argument("--output", "-o", help="Output folder")
    p.add_argument("--source", default="Japanese", help="Source language (default: Japanese)")
    p.add_argument("--target", default="English", help="Target language (default: English)")
    p.add_argument(
        "--ocr",
        choices=["auto", "manga-ocr", "paddleocr-vl-1.6"],
        default="auto",
        help="auto = manga-ocr for Japanese, PaddleOCR-VL 1.6 otherwise",
    )
    p.add_argument(
        "--inpaint",
        default="lama_manga",
        choices=["lama_manga", "opencv", "none"],
        help="Inpainter for text outside bubbles / colored bubbles: lama_manga (default), "
        "opencv (simple fill) or none",
    )
    p.add_argument(
        "--osb",
        choices=["auto", "on", "off"],
        default="auto",
        help="Translate text outside speech bubbles (narration/SFX). Needs the gated "
        "deepghs/AnimeText_yolo model; auto = on if available",
    )
    p.add_argument("--font", default="Comic Neue", help="Font pack in ./fonts for bubble text")
    p.add_argument("--osb-font", default="Bangers", help="Font pack in ./fonts for outside-bubble text")
    p.add_argument(
        "--reading-direction",
        choices=["rtl", "ltr"],
        default=None,
        help="Default: rtl for Japanese, ltr otherwise",
    )
    p.add_argument(
        "--context-pages",
        type=int,
        default=3,
        help="Previous pages' text given to the LLM as story context (default: 3)",
    )
    p.add_argument(
        "--glossary",
        default=None,
        help="Glossary file to reuse/update for consistent names (default: <output>/glossary.txt; "
        "pass the previous chapter's to carry names over)",
    )
    p.add_argument("--instructions", default=None, help="Extra instructions for the translator LLM")
    p.add_argument(
        "--honorifics",
        choices=["drop", "keep"],
        default="drop",
        help="Japanese honorifics (-san, -sama, ...): drop them like a professional localization "
        "(default) or keep them",
    )
    p.add_argument(
        "--sfx",
        choices=["label", "redraw"],
        default="label",
        help="Sound effects outside bubbles: keep the artist's lettering and add a small English "
        "label beside it, like a scanlation (default), or erase it and draw the English instead",
    )
    p.add_argument(
        "--adult",
        action="store_true",
        help="18+ content: translate explicit lines fully instead of softening them",
    )
    p.add_argument(
        "--keep-other-language",
        action="store_true",
        help="Don't erase blocks in another script (e.g. a Chinese translation printed beside the Japanese)",
    )
    p.add_argument(
        "--vision",
        nargs="?",
        const="full",
        default=None,
        choices=["off", "short", "full"],
        help="Show the translator the page image. short (default): translate from text, then redo "
        "only short items (sound effects, gasps, one-word lines, names read from furigana) with the "
        "page in view; full (bare --vision): the page for every line, which also adds meaning errors "
        "and guesses gender from looks; off. Loads the model's vision projector (~2 GB more VRAM). "
        "Default off with --model 12b (no VRAM left on 12 GB), --gguf and --llm-url",
    )
    p.add_argument(
        "--no-ocr-correction",
        dest="ocr_correction",
        action="store_false",
        help="Don't let the LLM fix lines that OCR obviously misread",
    )
    p.add_argument("--format", choices=["auto", "png", "jpeg"], default="png", help="Output format")
    p.add_argument(
        "--vertical-text",
        action="store_true",
        help="Allow stacking short text vertically in tall bubbles (default: on only for CJK targets)",
    )

    g = p.add_argument_group("LLM server")
    g.add_argument(
        "--llm-url",
        default=None,
        help="Use an already-running OpenAI-compatible server instead of starting llama-server",
    )
    g.add_argument("--llm-model-name", default=None, help="Model name to request from --llm-url")
    g.add_argument(
        "--model",
        choices=list(presets.MODEL_PRESETS),
        default="26b",
        help="Translator: 26b (default, best, ~2.4 GB VRAM) or 12b (faster, all on GPU)",
    )
    g.add_argument("--gguf", default=None, help="Custom GGUF weights for llama-server (overrides --model)")
    g.add_argument("--mmproj", default=None, help="Vision projector (for --vision)")
    g.add_argument("--port", type=int, default=8080)
    g.add_argument("--ctx", type=int, default=16384, help="LLM context size")
    g.add_argument(
        "--llm-server-args",
        default="",
        metavar="ARGS",
        help='Extra llama-server flags, e.g. "--fit on --fit-target 4500 -ctk q8_0 -ctv q8_0"',
    )
    g.add_argument(
        "--cpu-moe",
        type=int,
        default=None,
        metavar="N",
        help="MoE models only: keep experts of the first N layers in system RAM to save VRAM",
    )

    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Folder input: skip pages that already have a translated output (resume)",
    )
    p.add_argument(
        "--skip-existing-since",
        type=float,
        default=0.0,
        metavar="UNIX_TIME",
        help="With --skip-existing: only trust outputs written after this time",
    )
    p.add_argument(
        "--offline", action="store_true", help="Forbid all network access (models must already be downloaded)"
    )
    p.add_argument("--verbose", action="store_true", help="Detailed pipeline log")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Any setting, e.g. --set typesetting.bubbles.max_font_size=28 (repeatable)",
    )
    p.add_argument("--show-settings", action="store_true", help="Print the run's settings as JSON and exit")
    p.add_argument("--dry-run", action="store_true", help="Print the pipeline commands and exit")

    d = p.add_argument_group("evaluation (eval/harness.py)")
    d.add_argument("--llm-cache", default=None, metavar="DIR", help="Store / replay LLM answers by request")
    d.add_argument(
        "--llm-fresh",
        action="store_true",
        help="cache_prompt=false on every request: repeatable temperature-0 output",
    )
    d.add_argument(
        "--llm-replay",
        action="store_true",
        help="Answers only from --llm-cache; no LLM server; misses fail at once",
    )
    d.add_argument("--llm-misses", default=None, metavar="FILE", help="With --llm-replay: log misses here")
    d.add_argument("--dump-text", default=None, metavar="DIR", help="Write each page's regions as JSON")
    return p.parse_args()


# --------------------------------------------------------------------------- main


def build_settings(args: argparse.Namespace, out: Path, batch: bool, outside_text: bool) -> Settings:
    settings = presets.build(
        batch=batch,
        output=out,
        outside_text=outside_text,
        source=args.source,
        target=args.target,
        ocr=args.ocr,
        inpaint=args.inpaint,
        font=args.font,
        osb_font=args.osb_font,
        reading_direction=args.reading_direction,
        context_pages=args.context_pages,
        glossary=args.glossary,
        instructions=args.instructions,
        honorifics=args.honorifics,
        sfx=args.sfx,
        adult=args.adult,
        keep_other_language=args.keep_other_language,
        vision=args.vision,
        ocr_correction=args.ocr_correction,
        output_format=args.format,
        vertical_text=args.vertical_text,
        llm_url=args.llm_url,
        llm_model_name=args.llm_model_name,
        model=args.model,
        gguf=args.gguf,
        mmproj=args.mmproj,
        port=args.port,
        ctx=args.ctx,
        llm_server_args=args.llm_server_args,
        cpu_moe=args.cpu_moe,
        verbose=args.verbose,
        log=log,
    )
    d = settings.debug
    d.llm_cache = Path(args.llm_cache).resolve() if args.llm_cache else None
    d.llm_fresh, d.llm_replay = args.llm_fresh, args.llm_replay
    d.llm_misses = Path(args.llm_misses).resolve() if args.llm_misses else None
    d.dump_text = Path(args.dump_text).resolve() if args.dump_text else None
    try:
        return presets.apply_overrides(settings, args.set)
    except (KeyError, ValueError) as e:
        sys.exit(f"--set: {e}")


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    # No telemetry, ever. Downloads (first run only) are still allowed unless --offline.
    env.update(
        HF_HUB_DISABLE_TELEMETRY="1",
        YOLO_OFFLINE="1",
        DO_NOT_TRACK="1",
        PYTHONUNBUFFERED="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    if not args.input or not args.output:
        sys.exit("error: --input and --output are required")
    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not inp.exists():
        sys.exit(f"Input not found: {inp}")

    if args.osb == "auto":
        osb = osb_model_available(args.offline)
        if not osb:
            log(
                "Outside-bubble text: OFF (gated model deepghs/AnimeText_yolo not accessible; "
                "accept its terms on huggingface.co to enable)"
            )
    else:
        osb = args.osb == "on"
    check_font_pack(args.font)
    if osb:
        check_font_pack(args.osb_font)

    batch = inp.is_dir() or inp.suffix.lower() in {".zip", ".cbz", ".txt"}
    settings = build_settings(args, out, batch, osb)
    if args.show_settings:
        print(settings.to_json())
        return

    t = settings.translation
    if not settings.server.launch:
        if not server_healthy(t.url):
            log(f"warning: {t.url} does not answer /health; continuing anyway")
    elif not settings.debug.llm_replay and server_healthy(t.url):
        sys.exit(
            f"Something is already listening on port {settings.server.port}. "
            f"Pass --llm-url {t.url} to reuse it, or --port to pick another."
        )

    # (input, output) per pipeline run. The pipeline only reads the top level of a folder, so a
    # volume folder with chapter subfolders becomes one run per chapter, mirrored in the output.
    fmt = settings.output.format
    jobs: list[tuple[Path, Path]] = []
    if inp.is_dir():
        dirs = image_dirs(inp)
        if not dirs:
            sys.exit(f"No images ({', '.join(sorted(IMAGE_EXTS))}) found in {inp}")
        jobs = [(d, out / d.relative_to(inp)) for d in dirs]
    elif batch:
        jobs = [(inp, out)]
    else:  # single image: the pipeline wants a file path
        ext = inp.suffix if fmt == "auto" else (".jpg" if fmt == "jpeg" else ".png")
        jobs = [(inp, out / f"{inp.stem}{ext}")]

    settings_file = out / "settings.json"
    pipeline = [sys.executable, "-m", "mangatl.cli", "--settings", str(settings_file)]  # run from ROOT
    if batch:
        pipeline.append("--batch")
    if args.dry_run:
        for src, dst in jobs:
            print(shlex.join(pipeline + ["--input", str(src), "--output", str(dst)]))
        return
    out.mkdir(parents=True, exist_ok=True)
    settings.save(settings_file)  # secrets excluded; the pipeline reads them from the environment

    if settings.server.launch and not settings.debug.llm_replay:
        start_llama_server(
            settings.server, t.vision != "off", presets.MODEL_PRESETS[settings.server.model]["repo"]
        )

    llm = Path(settings.server.gguf).stem if settings.server.launch else t.model
    log(
        f"{inp} -> {out}  ({t.source_language} -> {t.target_language}, LLM: {llm}, "
        f"OCR: {settings.ocr.method}, inpaint: {settings.cleaning.inpainter}, "
        f"OSB: {'on' if osb else 'off'}, page image: {t.vision})"
    )
    t0 = time.time()
    # The pipeline logs model-loading problems but still exits 0 (and writes untranslated
    # pages), so watch its output and turn those into a failing exit code.
    fatal_markers = (
        "Error loading",
        "Failed to download",
        "Error during detection",
        "Failed image paths saved to",
    )
    problems: list[str] = []
    # A CUDA fault poisons the whole process ("sticky" error): every later page would come out
    # untranslated. Stop at once, drop the page in progress and resume in a fresh process.
    sticky_markers = ("illegal memory access", "Sticky error detected", "device-side assert")
    max_gpu_restarts = 2
    rc = 0
    tmp_dirs: list[tempfile.TemporaryDirectory] = []
    for n, (src0, dst) in enumerate(jobs, 1):
        if len(jobs) > 1:
            log(f"[{n}/{len(jobs)}] {src0.relative_to(inp) if src0 != inp else '.'}")
        if batch:
            dst.mkdir(parents=True, exist_ok=True)
        skip_existing, restarts = args.skip_existing, 0
        while True:
            src = src0
            if skip_existing and src0.is_dir():
                pages = [f for f in src0.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
                # --skip-existing (resume): trust outputs since --skip-existing-since (default: any).
                # Restart after a GPU fault in a fresh run: only outputs written by this run.
                since = args.skip_existing_since if args.skip_existing else t0

                def finished(f: Path) -> bool:
                    out_file = dst / translated_name(f, fmt)
                    return out_file.exists() and out_file.stat().st_mtime >= since

                todo = [f for f in pages if not finished(f)]
                if not todo:
                    log("  already translated, skipping")
                    break
                if len(todo) < len(pages):  # run only the missing pages, via a folder of symlinks
                    log(f"  resuming: {len(todo)}/{len(pages)} pages left")
                    tmp = tempfile.TemporaryDirectory(prefix="mt_resume_")
                    tmp_dirs.append(tmp)
                    for f in todo:
                        (Path(tmp.name) / f.name).symlink_to(f)
                    src = Path(tmp.name)
            full = pipeline + ["--input", str(src), "--output", str(dst)]
            proc = subprocess.Popen(
                full,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            current, sticky = None, None
            for line in proc.stdout:
                if line.startswith("Fontconfig "):  # harmless noise from skia vs. system fontconfig
                    continue
                sys.stdout.write(line)
                m = re.match(
                    r"(?:Processing|Retrying) \d+/\d+: (.+?)(?: \(warming up models\))?$", line.strip()
                )
                if m:
                    current = m.group(1)
                if any(k in line for k in sticky_markers):
                    sticky = line.strip()
                    proc.terminate()
                    break
                if any(k in line for k in fatal_markers):
                    problems.append(line.strip())
            if sticky:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                if current:  # its output (if any) was made with a broken GPU context
                    (dst / Path(current).parent / translated_name(Path(current), fmt)).unlink(missing_ok=True)
                if restarts < max_gpu_restarts:
                    restarts += 1
                    skip_existing = True
                    log(
                        f"GPU error on {current or 'a page'} ({sticky[:90]}); "
                        f"resuming in a fresh process ({restarts}/{max_gpu_restarts})"
                    )
                    continue
                problems.append(f"GPU error, gave up after {max_gpu_restarts} restarts: {sticky}")
                rc = rc or 3
                break
            rc = proc.wait() or rc
            break
    if rc == 0 and problems:
        rc = 2
        log("Pipeline reported problems (output may be incomplete):")
        for line in problems[:10]:
            log(f"  {line[:200]}")
    log(f"Finished in {time.time() - t0:.0f}s (exit code {rc})")
    sys.exit(rc)


if __name__ == "__main__":
    # Turn SIGTERM (e.g. "cancel" in the web UI) into a normal exit so llama-server is stopped too
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if shutil.which("nvidia-smi") is None:
        print("[translate] warning: no NVIDIA GPU tools found; this will be very slow", file=sys.stderr)
    main()
