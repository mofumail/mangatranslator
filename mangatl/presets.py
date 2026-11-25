"""Presets: build the Settings for a run from the user-facing options (translate.py, web UI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mangatl.settings import ROOT, Settings

LLM_DIR = ROOT / "models" / "llm"
# Translator presets (Gemma 4 QAT GGUFs from unsloth). Measured on 10 OpenMantra pages, RTX 4070 Ti:
#   26b: MoE 26B-A4B, experts in system RAM -> ~2.4 GB VRAM, ~10 s/page, best translations
#   12b: dense 12B fully on GPU            -> ~7.4 GB VRAM,  ~6 s/page, a few more meaning errors
MODEL_PRESETS = {
    "26b": dict(
        gguf=LLM_DIR / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        cpu_moe=99,
        mmproj=LLM_DIR / "gemma-4-26B-A4B-mmproj-F16.gguf",
        repo="unsloth/gemma-4-26B-A4B-it-qat-GGUF",
    ),
    "12b": dict(
        gguf=LLM_DIR / "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        cpu_moe=0,
        mmproj=LLM_DIR / "mmproj-F16.gguf",
        repo="unsloth/gemma-4-12B-it-qat-GGUF",
    ),
}
CJK_LANGUAGES = ("chinese", "japanese", "korean")


def model_alias(gguf: str | Path) -> str:
    """Served model name, unique per GGUF so translation caches never mix models.
    Avoids the substring "gemma-", which the model tables reserve for Google API models."""
    return "local-" + Path(gguf).stem.lower().replace("gemma-", "gemma")


def build(
    *,
    batch: bool,
    output: Path,
    outside_text: bool,
    source: str = "Japanese",
    target: str = "English",
    ocr: str = "auto",
    inpaint: str = "lama_manga",
    font: str = "Comic Neue",
    osb_font: str = "Bangers",
    reading_direction: str | None = None,
    context_pages: int = 3,
    glossary: str | None = None,
    instructions: str | None = None,
    honorifics: str = "drop",
    sfx: str = "label",
    adult: bool = False,
    keep_other_language: bool = False,
    vision: str | None = None,
    ocr_correction: bool = True,
    output_format: str = "png",
    vertical_text: bool = False,
    llm_url: str | None = None,
    llm_model_name: str | None = None,
    model: str = "26b",
    gguf: str | None = None,
    mmproj: str | None = None,
    port: int = 8080,
    ctx: int = 16384,
    llm_server_args: str = "",
    cpu_moe: int | None = None,
    verbose: bool = False,
    log=print,
) -> Settings:
    """Settings for a translate.py run. `batch` (folder / archive input) turns on the glossary
    and previous-page context; `outside_text` says whether the outside-text detector is usable."""
    preset = MODEL_PRESETS[model]
    # short vision by default only where it is measured to fit: the 26B with experts in RAM.
    # The 12B is all on the GPU; with the projector the page runs out of VRAM (12 GB card).
    vision_fits = gguf is None and model == "26b" and not llm_url
    if gguf is None:
        gguf = str(preset["gguf"])
        cpu_moe = preset["cpu_moe"] if cpu_moe is None else cpu_moe
    mmproj = mmproj or str(preset["mmproj"])
    if vision is None:
        vision = "short" if vision_fits else "off"
        if vision != "off" and not Path(mmproj).exists():
            log(f"Vision projector not found ({mmproj}); translating without the page image")
            vision = "off"
    japanese = source.lower() == "japanese"
    if ocr == "auto":
        ocr = "manga-ocr" if japanese else "paddleocr-vl-1.6"
    cjk_target = any(k in target.lower() for k in CJK_LANGUAGES)

    s = Settings()
    s.runtime.verbose = verbose
    s.server.launch = not llm_url
    s.server.model, s.server.gguf, s.server.mmproj = model, gguf, mmproj
    s.server.port, s.server.ctx, s.server.cpu_moe, s.server.extra_args = (
        port,
        ctx,
        cpu_moe or 0,
        llm_server_args,
    )

    s.detection.outside_text.enabled = outside_text
    s.detection.outside_text.cover_with_bubbles = outside_text
    s.ocr.method = ocr
    s.ocr.keep_other_language = keep_other_language
    s.cleaning.inpainter = inpaint

    t = s.translation
    t.url = llm_url or f"http://127.0.0.1:{port}/v1"
    t.model = (llm_model_name or "default") if llm_url else model_alias(gguf)
    t.source_language, t.target_language = source, target
    t.reading_direction = reading_direction or ("rtl" if japanese else "ltr")
    t.context_pages = context_pages if batch else 0
    t.glossary = batch  # names/terms memory, shared by all chapters of a run
    t.glossary_path = (
        (Path(glossary).expanduser().resolve() if glossary else output / "glossary.txt") if batch else None
    )
    t.instructions, t.honorifics, t.adult = instructions, honorifics, adult
    t.ocr_correction, t.vision = ocr_correction, vision

    s.typesetting.bubbles.font = font
    s.typesetting.outside.font = osb_font
    # letter-stacked Latin text ("H/u/h/?") reads badly in bubbles; outside them it is allowed
    # when it makes text much bigger (text_renderer._auto_vertical_layout_is_better)
    s.typesetting.bubbles.auto_vertical = vertical_text or cjk_target
    s.typesetting.sfx = sfx
    s.output.format = output_format
    return s


def apply_overrides(settings: Settings, assignments: list[str]) -> Settings:
    """--set section.field=value (value parsed as JSON when it can be, else as a string)."""
    import json

    changes: dict[str, Any] = {}
    for a in assignments:
        if "=" not in a:
            raise ValueError(f"--set expects path=value, got {a!r}")
        path, raw = a.split("=", 1)
        try:
            changes[path.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            changes[path.strip()] = raw
    return settings.updated(**changes).validated() if changes else settings
