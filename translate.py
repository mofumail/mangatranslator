#!/usr/bin/env python3
"""Fully local batch manga translator.

Pipeline (all on this machine):
  detection   YOLO speech-bubble seg + RT-DETR conjoined bubbles + panel YOLO (reading order)
  OCR         manga-ocr (Japanese) or PaddleOCR-VL 1.6 (other languages), or the LLM itself
  cleaning    flat-fill for bubbles, SAM-TS-L glyph masks + LaMa-manga inpainting for the rest
  translate   Gemma 4 12B (QAT, Q4_K_XL) served by a local llama.cpp server
  typeset     MangaTranslator's renderer (auto font size, hyphenation, vertical stacking)

The heavy lifting is done by MangaTranslator (./MangaTranslator, upstream checkout).
This wrapper starts/stops llama-server, picks sane local defaults and runs a batch.

Usage:
  uv run python translate.py --input chapter01/ --output chapter01_en/
  uv run python translate.py --input vol1.cbz --output vol1_en/ --target English
Unknown flags are passed straight through to MangaTranslator (see `--upstream-help`).
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
UPSTREAM = ROOT / "MangaTranslator"
MODELS_DIR = ROOT / "models"
FONTS_DIR = ROOT / "fonts"
LOG_DIR = ROOT / "logs"
LLAMA_SERVER = ROOT / "bin" / "llama.cpp" / "llama-server"
LLM_DIR = MODELS_DIR / "llm"
# Translator presets (Gemma 4 QAT GGUFs from unsloth). Measured on 10 OpenMantra pages, RTX 4070 Ti:
#   26b: MoE 26B-A4B, experts in system RAM -> ~2.4 GB VRAM, ~10 s/page, best translations
#   12b: dense 12B fully on GPU            -> ~7.4 GB VRAM,  ~6 s/page, a few more meaning errors
MODEL_PRESETS = {
    "26b": dict(gguf=LLM_DIR / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", cpu_moe=99,
                mmproj=LLM_DIR / "gemma-4-26B-A4B-mmproj-F16.gguf",
                repo="unsloth/gemma-4-26B-A4B-it-qat-GGUF"),
    "12b": dict(gguf=LLM_DIR / "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf", cpu_moe=0,
                mmproj=LLM_DIR / "mmproj-F16.gguf",
                repo="unsloth/gemma-4-12B-it-qat-GGUF"),
}


def model_alias(gguf: str) -> str:
    """Served model name, unique per GGUF so translation caches never mix models.
    Avoids the substring "gemma-", which upstream reserves for Google API models."""
    return "local-" + Path(gguf).stem.lower().replace("gemma-", "gemma")

OSB_TEXT_MODEL = MODELS_DIR / "detection" / "yolo" / "animetext_yolov12x.pt"
OSB_TEXT_URL = (
    "https://huggingface.co/deepghs/AnimeText_yolo/resolve/main/yolo12x_animetext/model.pt"
)


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


def start_llama_server(args: argparse.Namespace, vision: bool) -> subprocess.Popen:
    if not LLAMA_SERVER.exists():
        sys.exit(f"llama-server not found at {LLAMA_SERVER}")
    if not Path(args.gguf).exists():
        sys.exit(f"LLM weights not found: {args.gguf}")

    cmd = [
        str(LLAMA_SERVER),
        "-m", str(args.gguf),
        "--alias", model_alias(args.gguf),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "-c", str(args.ctx),
        "-np", "1",
        "-ngl", "99",
        "-fa", "on",
        "--reasoning", "off",  # thinking burns tokens without improving translations
        "--no-webui",
    ]
    if vision:
        if not Path(args.mmproj).exists():
            sys.exit(f"Vision projector not found: {args.mmproj}\n"
                     f"Download mmproj-F16.gguf from https://huggingface.co/{args.repo} to that path.")
        # Gemma 4 vision needs the whole image in one ubatch
        cmd += ["--mmproj", str(args.mmproj), "-b", "2048", "-ub", "2048"]
    else:
        cmd += ["--no-mmproj"]
    if args.cpu_moe:  # MoE models: keep expert weights of the first N layers in system RAM
        cmd += ["--n-cpu-moe", str(args.cpu_moe)]
    extra = shlex.split(args.llm_server_args)
    if "--fit" in extra or "-fit" in extra:  # let llama.cpp choose the GPU layer count
        i = cmd.index("-ngl")
        del cmd[i:i + 2]
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

    url = f"http://127.0.0.1:{args.port}/v1"
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


def font_pack(name: str) -> str:
    p = Path(name)
    if not p.is_absolute():
        p = FONTS_DIR / name
    if not p.is_dir() or not (list(p.glob("*.ttf")) + list(p.glob("*.otf"))):
        packs = ", ".join(sorted(d.name for d in FONTS_DIR.iterdir() if d.is_dir()))
        sys.exit(f"Font pack '{name}' not found (folder of .ttf/.otf). Available: {packs}")
    return str(p)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def natural_key(path: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def image_dirs(root: Path) -> list[Path]:
    """root and every subfolder that directly contains images, in natural (chapter) order."""
    dirs = [d for d in [root, *root.rglob("*")] if d.is_dir()
            and any(f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in d.iterdir())]
    return sorted(dirs, key=lambda d: natural_key(d.relative_to(root)))


def translated_name(img: Path, fmt: str) -> str:
    ext = img.suffix if fmt == "auto" else (".jpg" if fmt == "jpeg" else ".png")
    return f"{img.stem}_translated{ext}"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Fully local batch manga translator (MangaTranslator + llama.cpp).",
        epilog="Any other flags are forwarded to MangaTranslator; see --upstream-help.",
    )
    p.add_argument("--input", "-i", help="Folder of images, a single image, or a .zip/.cbz archive")
    p.add_argument("--output", "-o", help="Output folder")
    p.add_argument("--source", default="Japanese", help="Source language (default: Japanese)")
    p.add_argument("--target", default="English", help="Target language (default: English)")
    p.add_argument(
        "--ocr",
        choices=["auto", "manga-ocr", "paddleocr-vl-1.6", "llm"],
        default="auto",
        help="auto = manga-ocr for Japanese, PaddleOCR-VL 1.6 otherwise. "
        "'llm' lets Gemma read the bubbles itself (vision, one-step; slower)",
    )
    p.add_argument(
        "--inpaint",
        default="lama_manga",
        help="Inpainter for text outside bubbles / colored bubbles: lama_manga (default, light), "
        "flux_klein_4b, flux_klein_9b, opencv, ... (FLUX needs more VRAM than is free next to the LLM)",
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
    p.add_argument("--reading-direction", choices=["rtl", "ltr"], default=None,
                   help="Default: rtl for Japanese, ltr otherwise")
    p.add_argument("--context-pages", type=int, default=3,
                   help="Previous pages' text given to the LLM as story context (default: 3)")
    p.add_argument("--glossary", default=None,
                   help="Glossary file to reuse/update for consistent names (default: <output>/glossary.txt; "
                   "pass the previous chapter's to carry names over)")
    p.add_argument("--instructions", default=None, help="Extra instructions for the translator LLM")
    p.add_argument("--adult", action="store_true",
                   help="18+ content: translate explicit lines fully instead of softening them")
    p.add_argument("--keep-other-language", action="store_true",
                   help="Don't erase blocks in another script (e.g. a Chinese translation printed beside the Japanese)")
    p.add_argument("--vision", action="store_true",
                   help="Show the translator the whole page too (who speaks, what happens; catches OCR "
                   "misreads). Loads the model's vision projector: ~2 GB more VRAM, slower")
    p.add_argument("--no-ocr-correction", dest="ocr_correction", action="store_false",
                   help="Don't let the LLM fix lines that OCR obviously misread")
    p.add_argument("--format", choices=["auto", "png", "jpeg"], default="png", help="Output format")
    p.add_argument("--vertical-text", action="store_true",
                   help="Allow stacking short text vertically in tall bubbles (default: on only for CJK targets)")

    g = p.add_argument_group("LLM server")
    g.add_argument("--llm-url", default=None,
                   help="Use an already-running OpenAI-compatible server instead of starting llama-server")
    g.add_argument("--llm-model-name", default=None, help="Model name to request from --llm-url")
    g.add_argument("--model", choices=list(MODEL_PRESETS), default="26b",
                   help="Translator: 26b (default, best, ~2.4 GB VRAM) or 12b (faster, all on GPU)")
    g.add_argument("--gguf", default=None, help="Custom GGUF weights for llama-server (overrides --model)")
    g.add_argument("--mmproj", default=None, help="Vision projector (for --vision / --ocr llm)")
    g.add_argument("--port", type=int, default=8080)
    g.add_argument("--ctx", type=int, default=16384, help="LLM context size")
    g.add_argument("--llm-server-args", default="", metavar="ARGS",
                   help='Extra llama-server flags, e.g. "--fit on --fit-target 4500 -ctk q8_0 -ctv q8_0"')
    g.add_argument("--cpu-moe", type=int, default=None, metavar="N",
                   help="MoE models only: keep experts of the first N layers in system RAM to save VRAM")

    p.add_argument("--skip-existing", action="store_true",
                   help="Folder input: skip pages that already have a translated output (resume)")
    p.add_argument("--offline", action="store_true",
                   help="Forbid all network access (models must already be downloaded)")
    p.add_argument("--dry-run", action="store_true", help="Print the upstream command and exit")
    p.add_argument("--upstream-help", action="store_true", help="Show all MangaTranslator options")
    return p.parse_known_args()


# --------------------------------------------------------------------------- main


def main() -> None:
    args, passthrough = parse_args()
    preset = MODEL_PRESETS[args.model]
    args.repo = preset["repo"]
    if args.gguf is None:
        args.gguf = str(preset["gguf"])
        args.cpu_moe = preset["cpu_moe"] if args.cpu_moe is None else args.cpu_moe
    args.mmproj = args.mmproj or str(preset["mmproj"])
    args.cpu_moe = args.cpu_moe or 0
    env = os.environ.copy()
    # No telemetry, ever. Downloads (first run only) are still allowed unless --offline.
    env.update(HF_HUB_DISABLE_TELEMETRY="1", YOLO_OFFLINE="1", DO_NOT_TRACK="1",
               GRADIO_ANALYTICS_ENABLED="False", PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.keep_other_language:
        env["MT_KEEP_OTHER_LANGUAGE"] = "1"

    upstream_main = [sys.executable, str(ROOT / "mt_upstream.py")]  # upstream CLI + local OCR patch
    if args.upstream_help:
        sys.exit(subprocess.call([sys.executable, str(UPSTREAM / "main.py"), "--help"], cwd=UPSTREAM, env=env))
    if not args.input or not args.output:
        sys.exit("error: --input and --output are required")

    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not inp.exists():
        sys.exit(f"Input not found: {inp}")
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    ocr = args.ocr
    if ocr == "auto":
        ocr = "manga-ocr" if args.source.lower() == "japanese" else "paddleocr-vl-1.6"
    vision = ocr == "llm"  # one-step: Gemma reads the text itself
    if args.vision and not vision:
        env["MT_PAGE_IMAGE"] = "1"  # local OCR, plus the page image for the translator
    reading_dir = args.reading_direction or ("rtl" if args.source.lower() == "japanese" else "ltr")

    if args.osb == "auto":
        osb = osb_model_available(args.offline)
        if not osb:
            log("Outside-bubble text: OFF (gated model deepghs/AnimeText_yolo not accessible; "
                "accept its terms on huggingface.co to enable)")
    else:
        osb = args.osb == "on"

    # LLM endpoint
    if args.llm_url:
        llm_url, model_name = args.llm_url, args.llm_model_name or "default"
        if not server_healthy(llm_url):
            log(f"warning: {llm_url} does not answer /health; continuing anyway")
    else:
        llm_url, model_name = f"http://127.0.0.1:{args.port}/v1", model_alias(args.gguf)
        if server_healthy(llm_url):
            sys.exit(f"Something is already listening on port {args.port}. "
                     f"Pass --llm-url {llm_url} to reuse it, or --port to pick another.")

    batch = inp.is_dir() or inp.suffix.lower() in {".zip", ".cbz", ".txt"}
    # (input, output) per upstream run. Upstream only reads the top level of a folder, so a
    # volume folder with chapter subfolders becomes one run per chapter, mirrored in the output.
    jobs: list[tuple[Path, Path]] = []
    if inp.is_dir():
        dirs = image_dirs(inp)
        if not dirs:
            sys.exit(f"No images ({', '.join(sorted(IMAGE_EXTS))}) found in {inp}")
        jobs = [(d, out / d.relative_to(inp)) for d in dirs]
    elif batch:
        jobs = [(inp, out)]
    else:  # single image: upstream wants a file path
        ext = inp.suffix if args.format == "auto" else (".jpg" if args.format == "jpeg" else ".png")
        jobs = [(inp, out / f"{inp.stem}{ext}")]

    cmd = [
        "--batch" if batch else "",
        "--provider", "OpenAI-Compatible",
        "--openai-compatible-url", llm_url,
        "--model-name", model_name,
        "--input-language", args.source,
        "--output-language", args.target,
        "--reading-direction", reading_dir,
        "--models", str(MODELS_DIR),
        "--font-dir", font_pack(args.font),
        "--output-format", args.format,
        "--batch-previous-context-texts", str(args.context_pages),
        "--osb-inpainting-method", args.inpaint,
        # narrow/tall regions (vertical Japanese asides) may grow so English fits
        "--osb-render-expansion-narrow", "2.0",
    ]
    if vision:
        cmd += ["--translation-mode", "one-step", "--ocr-method", "LLM"]
    else:
        cmd += ["--translation-mode", "two-step", "--ocr-method", ocr]
    if args.adult:
        cmd += ["--ero-doujinshi-mode"]
    if args.ocr_correction and not vision:
        cmd += ["--ocr-correction"]
    if osb:
        cmd += ["--osb-enable", "--osb-font-dir", font_pack(args.osb_font)]
    else:
        cmd += ["--no-osb-text-verification"]
    if batch:  # names/terms memory, shared by all chapters; reuse it later with --glossary
        glossary = Path(args.glossary).expanduser().resolve() if args.glossary else out / "glossary.txt"
        cmd += ["--batch-glossary-file", str(glossary)]
    cjk_target = any(k in args.target.lower() for k in ("chinese", "japanese", "korean"))
    if not (args.vertical_text or cjk_target):
        # letter-stacked Latin text ("H/u/h/?") reads badly in bubbles; outside them it's
        # allowed when it makes text much bigger (osb_ocr.install_vertical_only_if_much_bigger)
        cmd += ["--no-auto-vertical-text"]
    notes = [args.instructions] if args.instructions else []
    notes.append("Localize like a professional manga translator: natural, idiomatic English with contractions "
                 "and each character's own voice; keep the meaning exact but never translate word for word. "
                 "Narration reads like a light novel: past tense, flowing sentences.")
    # Outside-bubble fonts often lack italics, so emphasis markup must not replace quote marks
    notes.append("Render quoted speech in 「」 and 『』 with quotation marks (\"...\" and '...'); "
                 "never use * or ** markdown in place of quotation marks.")
    notes.append("An item with \" / \" separators is a list (chat comments, credits): translate each entry "
                 "and keep every \" / \" in place.")
    if batch:
        # Local models tend to write the English name on both sides of the new-terms block,
        # which upstream rejects ("source not on page"), leaving the glossary empty.
        notes.append(f"In the new-terms block, the left side of each line must be the term copied "
                     f"verbatim from the {args.source} transcription (original script), never its translation.")
    if notes:
        cmd += ["--special-instructions", "\n".join(notes)]
    cmd = [c for c in cmd if c] + passthrough

    if args.dry_run:
        for src, dst in jobs:
            full = upstream_main + ["--input", str(src), "--output", str(dst)] + cmd
            print(" ".join(f"'{c}'" if " " in c else c for c in full))
        return

    if not args.llm_url:
        start_llama_server(args, vision or args.vision)

    log(f"{inp} -> {out}  ({args.source} -> {args.target}, LLM: {Path(args.gguf).stem if not args.llm_url else model_name}, "
        f"OCR: {ocr}, inpaint: {args.inpaint}, "
        f"OSB: {'on' if osb else 'off'})")
    t0 = time.time()
    # Upstream logs model-loading problems but still exits 0 (and writes untranslated
    # pages), so watch its output and turn those into a failing exit code.
    fatal_markers = ("Error loading", "Failed to download", "Error during detection",
                     "Failed image paths saved to")
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
                # --skip-existing (resume): trust outputs since MT_SKIP_EXISTING_SINCE (default: any).
                # Restart after a GPU fault in a fresh run: only outputs written by this run.
                since = (float(os.environ.get("MT_SKIP_EXISTING_SINCE", "0") or 0)
                         if args.skip_existing else t0)

                def finished(f: Path) -> bool:
                    out_file = dst / translated_name(f, args.format)
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
            full = upstream_main + ["--input", str(src), "--output", str(dst)] + cmd
            proc = subprocess.Popen(full, cwd=UPSTREAM, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            current, sticky = None, None
            for line in proc.stdout:
                if line.startswith("Fontconfig "):  # harmless noise from skia vs. system fontconfig
                    continue
                sys.stdout.write(line)
                m = re.match(r"(?:Processing|Retrying) \d+/\d+: (.+?)(?: \(warming up models\))?$", line.strip())
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
                    (dst / Path(current).parent / translated_name(Path(current), args.format)).unlink(missing_ok=True)
                if restarts < max_gpu_restarts:
                    restarts += 1
                    skip_existing = True
                    log(f"GPU error on {current or 'a page'} ({sticky[:90]}); "
                        f"resuming in a fresh process ({restarts}/{max_gpu_restarts})")
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
