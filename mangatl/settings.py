"""Settings: one typed object that describes a translation run.

translate.py and the web UI build it (presets in `mangatl.presets`), it is written to
`<output>/settings.json` with every run (secrets excluded) and the pipeline reads nothing else:
no environment switches, no flag chains. Sections follow the pipeline stages.

Runtime-only state (the resolved torch device, the request coordinator for parallel requests,
the bubble detector weights path) lives in private attributes that are never serialised.
Secrets (Hugging Face token, API key) are resolved from the environment at load time and are
excluded from every dump.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = ROOT / "fonts"

# Labels error placeholders ("[OpenAI-Compatible: Missing item 3]"), which the renderer recognises
PROVIDER = "OpenAI-Compatible"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RuntimeSettings(_Section):
    """How the process runs; does not change what the pages look like."""

    models_dir: Path = ROOT / "models"
    device: Literal["auto", "cpu"] = "auto"
    hf_cache: bool = False  # load Hugging Face models from the default HF cache
    hf_token: str = Field(default="", exclude=True, repr=False)  # gated downloads; from HF_TOKEN / token file
    # VRAM the pipeline's models may hold (MB; None: whatever is free). Idle models are unloaded
    # to stay under it, e.g. to leave room for a game or a bigger translator.
    vram_budget_mb: float | None = Field(default=None, ge=0)
    verbose: bool = False
    cleaning_only: bool = False
    # Parallel requests to the translator (useful with hosted models; llama-server serves one)
    parallel_requests: int = Field(default=1, ge=1, le=20)
    parallel_within_pages: bool = False
    overlap_llm_with_inpaint: bool = True
    retry_failed_once: bool = True  # batch: retry failed pages once at the end


class ServerSettings(_Section):
    """The local llama-server translate.py starts (unless translation.url points elsewhere)."""

    launch: bool = True
    model: Literal["26b", "12b"] = "26b"
    gguf: str | None = None  # custom weights; overrides `model`
    mmproj: str | None = None  # vision projector
    port: int = 8080
    ctx: int = 16384
    cpu_moe: int | None = None  # MoE layers whose experts stay in RAM
    extra_args: str = ""


class OutsideTextDetection(_Section):
    """Text outside speech bubbles (narration, sound effects, signs)."""

    enabled: bool = True
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    text_free_only: bool = False  # use only the RT-DETR text_free class as source
    # Letters the text segmentation model finds outside bubbles and boxes, kept where the
    # detector confirms them on crops
    recover_missed_text: bool = True
    cover_with_bubbles: bool = True  # grow bubble boxes to fully cover detected text
    filter_page_numbers: bool = False
    page_filter_margin: float = Field(default=0.1, ge=0.0, le=0.3)
    page_filter_min_area: float = Field(default=0.05, ge=0.0, le=0.2)
    min_area_ignore_ratio: float = Field(default=0.0, ge=0.0, le=0.05)
    bbox_expansion_width: float = Field(default=0.1, ge=0.0, le=1.0)
    bbox_expansion_height: float = Field(default=0.1, ge=0.0, le=1.0)
    text_box_proximity_ratio: float = Field(default=0.02, ge=0.01, le=0.1)  # share of the shorter image side


class DetectionSettings(_Section):
    bubble_model: Literal["yolo_1", "yolo_2"] = "yolo_2"
    confidence: float = Field(default=0.6, ge=0.1, le=1.0)
    conjoined: bool = True  # RT-DETR detector for conjoined bubbles
    conjoined_confidence: float = Field(default=0.35, ge=0.1, le=1.0)
    panel_sorting: bool = True  # reading order by panels
    panel_confidence: float = Field(default=0.25, ge=0.05, le=1.0)
    outside_text: OutsideTextDetection = Field(default_factory=lambda: OutsideTextDetection())


class OcrSettings(_Section):
    method: Literal["manga-ocr", "paddleocr-vl-1.6"] = "manga-ocr"
    upscale: Literal["model", "model_lite", "lanczos", "none"] = "model_lite"  # small crops
    bubble_min_side: int = Field(default=128, ge=64, le=512)
    outside_min_side: int = Field(default=128, ge=64, le=512)
    whiteout_neighbors: bool = True  # blank neighbouring bubbles in conjoined crops
    # Pages that print a Chinese translation next to the Japanese: the Chinese block is erased
    # and not translated, unless this is set
    keep_other_language: bool = False


class CleaningSettings(_Section):
    threshold: int = Field(default=200, ge=0, le=255)
    otsu: bool = False
    auto_threshold: bool = True
    roi_shrink_px: int = Field(default=5, ge=0, le=10)
    # SAM-TS-L text segmentation; off, fills miss edge-hugging text and erase all ink enclosed
    # by paper (blush lines, frames), tilt and text colour fall back to ink heuristics
    text_segmentation: bool = True
    inpaint_colored_bubbles: bool = True  # only while outside-text detection is on
    inpainter: Literal["lama_manga", "opencv", "none"] = "lama_manga"
    luminance_correction: bool = True  # match inpainted patches to their surroundings
    seed: int = 1  # -1 = random


class TranslationSettings(_Section):
    url: str = "http://127.0.0.1:8080/v1"  # any OpenAI-compatible server
    api_key: str = Field(default="", exclude=True, repr=False)  # from OPENAI_COMPATIBLE_API_KEY
    model: str = "default"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=64)
    custom_sampling: bool = True
    max_tokens: int | None = Field(
        default=None, ge=4096, le=63488
    )  # None: 32768 for reasoning models, 4096 otherwise
    reasoning_effort: str | None = None
    effort: str | None = "medium"  # Claude models served through --llm-url
    verbosity: str | None = "low"  # GPT-5 / GPT-6 models served through --llm-url
    image_detail: str = "auto"
    source_language: str = "Japanese"
    target_language: str = "English"
    reading_direction: Literal["rtl", "ltr"] = "rtl"
    full_page_context: bool = True  # encode the page image for the translator
    context_image_max_side: int = Field(default=1024, ge=512, le=2560)
    context_pages: int = Field(default=3, ge=0, le=50)  # previous pages' text sent as context (batch)
    glossary: bool = True  # names/terms memory (batch)
    glossary_path: Path | None = None
    instructions: str | None = None  # the user's own instructions
    honorifics: Literal["drop", "keep"] = "drop"
    adult: bool = False  # 18+ content: translate explicit lines fully
    ocr_correction: bool = True  # let the LLM repair obvious OCR misreads
    force_cache: bool = False
    # Page image for the translator: "short" redoes only short items with the page in view,
    # "full" shows it for every line
    vision: Literal["off", "short", "full"] = "short"

    @property
    def special_instructions(self) -> str:
        """The instructions block of the prompt: the user's own plus the localization notes."""
        notes = [self.instructions] if self.instructions else []
        notes.append(
            "Localize like a professional manga translator: natural, idiomatic English with contractions "
            "and each character's own voice; keep the meaning exact but never translate word for word. "
            "Narration reads like a light novel: past tense, flowing sentences."
        )
        # Outside-bubble fonts often lack italics, so emphasis markup must not replace quote marks
        notes.append(
            "Render quoted speech in 「」 and 『』 with quotation marks (\"...\" and '...'); "
            "never use * or ** markdown in place of quotation marks."
        )
        notes.append(
            'An item with " / " separators is a list (chat comments, credits): translate each entry '
            'and keep every " / " in place.'
        )
        if self.honorifics == "drop":
            notes.append(
                "Drop Japanese honorifics (-san, -kun, -chan, -sama, -dono, -senpai). Where one carries "
                "meaning, say it the English way instead (Lady Priscilla, Master, big sis)."
            )
        else:
            notes.append("Keep Japanese honorifics on names as written (Leaf-san, Priscilla-sama).")
        notes.append(
            "A personal name in katakana that is a Western or fantasy name gets its natural Western "
            "spelling (プリシラ → Priscilla), not a syllable-by-syllable romanization."
        )
        # A bubble that is only エ / エエエ!! after 死ねェ is the same shout drawn out, not a new sound.
        # Guarded on the previous item so standalone あ/ん bubbles (moans, gasps) stay vocalizations.
        notes.append(
            "When an item is only a drawn-out vowel (エ, ェェ, エエエ!!, ーー) and the previous item ends "
            "on that same vowel, it is the previous word stretched across bubbles: repeat that English "
            "word drawn out (死ねェ / エ / エエエ!! → Die! / Die... / DIEEEE!!). Otherwise a lone vowel "
            "is a vocalization as usual."
        )
        if self.glossary:
            # Local models tend to write the English name on both sides of the new-terms block,
            # which is then rejected ("source not on page"), leaving the glossary empty.
            notes.append(
                f"In the new-terms block, the left side of each line must be the term copied "
                f"verbatim from the {self.source_language} transcription (original script), never its "
                f"translation. Sound effects and interjections are never terms."
            )
        return "\n".join(notes)


class TextStyle(_Section):
    """Lettering of one kind of text."""

    font: str  # font pack name in fonts/, or a folder path
    max_font_size: int
    min_font_size: int
    line_spacing: float = 1.0
    ligatures: bool = False
    padding: float = 4.0
    auto_vertical: bool = False  # allow stacking short text vertically
    vertical_line_spacing: float = 1.0
    vertical_font_size: float = 1.0

    @property
    def font_dir(self) -> str:
        p = Path(self.font)
        return str(p if p.is_dir() else FONTS_DIR / self.font)


class BubbleText(TextStyle):
    font: str = "Comic Neue"
    max_font_size: int = 32
    min_font_size: int = 8
    auto_font_size: bool = True  # size to the page's source lettering


class OutsideText(TextStyle):
    font: str = "Bangers"
    max_font_size: int = 64
    min_font_size: int = 10
    outline_ratio: float = 0.15  # outline around text on the art (and in colored bubbles)
    auto_vertical: bool = True
    # Narrow/tall or tiny regions (vertical Japanese asides) may grow so English fits
    expansion_narrow: float = Field(default=2.0, ge=1.0, le=3.0)
    expansion_tiny: float = Field(default=1.0, ge=1.0, le=3.0)
    expansion_aspect_threshold: float = Field(default=0.4, ge=0.05, le=1.0)
    expansion_area_threshold: float = Field(default=0.005, ge=0.0, le=0.05)
    follow_tilt: bool = True
    max_tilt: float = 20.0
    min_tilt: float = 3.0
    tilt_confidence: float = 0.5


class TypesettingSettings(_Section):
    bubbles: BubbleText = Field(default_factory=lambda: BubbleText())
    outside: OutsideText = Field(default_factory=lambda: OutsideText())
    hyphenate_before_scaling: bool = True
    hyphenation_min_word_length: int = Field(default=8, ge=4, le=10)
    badness_exponent: float = Field(default=3.0, ge=2.0, le=4.0)
    supersampling: int = Field(default=4, ge=1, le=16)
    detach_trailing_punctuation: bool = True
    # Sound effects: keep the artist's lettering with an English label beside it, or redraw
    sfx: Literal["label", "redraw"] = "label"


class OutputSettings(_Section):
    format: Literal["auto", "png", "jpeg"] = "png"
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    png_compression: int = Field(default=2, ge=0, le=6)
    # the project file (SQLite) recording every page's state and history, for the editor;
    # translate.py writes <output>/project.mtl
    project: Path | None = None


class PreprocessingSettings(_Section):
    upscale: bool = False  # upscale source pages before processing
    upscale_factor: float = Field(default=2.0, ge=1.0, le=8.0)
    upscale_model: Literal["model", "model_lite"] = "model_lite"
    auto_scale: bool = True  # scale kernels and font sizes to the page size (relative to 1 MP)
    processing_scale: float = 1.0  # the fixed scale when auto_scale is off
    split_long_strips: bool = True  # webtoons: translate section by section


class DebugSettings(_Section):
    """Evaluation aids (eval/harness.py)."""

    # LLM answers stored by request content and replayed for identical requests
    llm_cache: Path | None = None
    # cache_prompt=false on every request: llama.cpp evaluates each prompt anew instead of
    # reusing the previous request's prefix, which makes temperature-0 output repeatable
    llm_fresh: bool = False
    # answers only from llm_cache; a miss is logged to llm_misses and fails at once
    llm_replay: bool = False
    llm_misses: Path | None = None
    dump_text: Path | None = None  # <dir>/<page>.json: every region's box, OCR read, translation


class Settings(_Section):
    runtime: RuntimeSettings = Field(default_factory=lambda: RuntimeSettings())
    server: ServerSettings = Field(default_factory=lambda: ServerSettings())
    detection: DetectionSettings = Field(default_factory=lambda: DetectionSettings())
    ocr: OcrSettings = Field(default_factory=lambda: OcrSettings())
    cleaning: CleaningSettings = Field(default_factory=lambda: CleaningSettings())
    translation: TranslationSettings = Field(default_factory=lambda: TranslationSettings())
    typesetting: TypesettingSettings = Field(default_factory=lambda: TypesettingSettings())
    output: OutputSettings = Field(default_factory=lambda: OutputSettings())
    preprocessing: PreprocessingSettings = Field(default_factory=lambda: PreprocessingSettings())
    debug: DebugSettings = Field(default_factory=lambda: DebugSettings())

    _models: Any = PrivateAttr(default=None)
    _request_coordinator: Any = PrivateAttr(default=None)

    # --- runtime-only state -------------------------------------------------------------
    @property
    def verbose(self) -> bool:
        return self.runtime.verbose

    @property
    def models(self):
        """The engine's models this run calls (mangatl.engine.Models), attached by whoever runs
        it: with_models()."""
        if self._models is None:
            raise RuntimeError("no engine attached to these settings (Settings.with_models)")
        return self._models

    def with_models(self, models) -> Settings:
        """A copy whose pipeline calls go to `models` (an engine at a priority)."""
        new = self.model_copy()
        new._models, new._request_coordinator = models, self._request_coordinator
        return new

    @property
    def device(self):
        """The torch device the engine runs models on."""
        return self.models.device

    @property
    def request_coordinator(self):
        """Schedules parallel requests within a batch (runtime.parallel_requests > 1)."""
        return self._request_coordinator

    @request_coordinator.setter
    def request_coordinator(self, value) -> None:
        self._request_coordinator = value

    # --- persistence --------------------------------------------------------------------
    def resolve_secrets(self) -> Settings:
        """Fill secrets from the environment (they are never stored)."""
        if not self.runtime.hf_token:
            token = os.environ.get("HF_TOKEN", "")
            if not token:
                token_file = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
                if token_file.is_file():
                    token = token_file.read_text().strip()
            self.runtime.hf_token = token
        if not self.translation.api_key:
            self.translation.api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
        return self

    def to_json(self) -> str:
        return self.model_dump_json(indent=1)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json() + "\n")

    @classmethod
    def load(cls, path: Path | str) -> Settings:
        return cls.model_validate(json.loads(Path(path).read_text())).resolve_secrets()

    def updated(self, **changes: Any) -> Settings:
        """Copy with changes given as dotted paths, e.g. updated(**{"typesetting.sfx": "redraw"}).
        A cheap nested copy (runtime state carried over, values not re-validated); use
        validated() for values from users."""
        new: BaseModel = self
        for dotted, value in changes.items():
            new = _with_path(new, dotted.split("."), value, dotted)
        return cast(Settings, new)

    def validated(self) -> Settings:
        """Re-validate every value (e.g. after --set), keeping secrets and runtime state."""
        new = type(self).model_validate(self.model_dump())
        new.runtime.hf_token, new.translation.api_key = self.runtime.hf_token, self.translation.api_key
        new._models, new._request_coordinator = self._models, self._request_coordinator
        return new


def _with_path(model: BaseModel, path: list[str], value: Any, dotted: str) -> BaseModel:
    head, *rest = path
    if head not in type(model).model_fields:
        raise KeyError(f"unknown setting {dotted!r}")
    if rest:
        value = _with_path(getattr(model, head), rest, value, dotted)
    return model.model_copy(update={head: value})
