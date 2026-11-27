"""Every model the pipeline runs: where its weights live, how to fetch them, how to load them.

Weights are looked up locally first; the network is used only for files that are missing
(never to re-check files already on disk, which stalled every load when the Hub was
unreachable). A pinned Hugging Face revision is checked against the download metadata the Hub
client writes next to the files.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from mangatl.common.exceptions import ModelError
from mangatl.common.log import log_message


class Model(StrEnum):
    UPSCALE = "upscale"  # 2x-AnimeSharpV4 RCAN (small OCR crops)
    UPSCALE_LITE = "upscale_lite"  # its fast variant
    BUBBLES = "bubbles"  # YOLOv8m speech-bubble segmentation
    BUBBLES_2 = "bubbles_2"  # Manga109 bubble segmentation (YOLO)
    CONJOINED = "conjoined"  # RT-DETR-v2 comic text and bubble detector (conjoined bubbles)
    OUTSIDE_TEXT = "outside_text"  # AnimeText YOLOv12x (text on the art)
    PANELS = "panels"  # Manga109 YOLOv11 panels (reading order)
    MANGA_OCR = "manga_ocr"
    PADDLE_OCR_VL = "paddle_ocr_vl"  # non-Japanese sources
    HAYAI = "hayai"  # Hayai OCR v2.5 Nova
    TEXT_LINES = "text_lines"  # comic-text-detector (ONNX, CPU)
    LAMA = "lama"  # LaMa manga inpainting
    TEXT_SAM = "text_sam"  # SAM-TS-L text segmentation (glyph masks)


HAYAI_REVISION = "2803d0dc005fa4396ab1e7c5a472d2c450a247f7"  # matches vendor/hayai
CTD_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
)
CTD_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"
_COMMIT = re.compile(r"[0-9a-f]{40}")


class Fetcher:
    """Finds a model's files under models_dir, downloading only what is missing."""

    def __init__(self, models_dir: Path, token: str | None, hf_cache: bool, offline: bool):
        self.models_dir = Path(models_dir).resolve()
        self.token = token or None
        self.hf_cache = hf_cache  # keep downloads in the Hugging Face cache instead
        self.offline = offline

    def _need_network(self, what: str) -> None:
        if self.offline:
            raise ModelError(f"{what} is not downloaded (offline)")
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    def hf_file(self, repo: str, filename: str, target: Path) -> Path:
        if not self.hf_cache and target.exists():
            return target
        if not self.hf_cache:
            self._need_network(target.name)
        from huggingface_hub import hf_hub_download

        if self.hf_cache:
            with contextlib.suppress(Exception):
                return Path(hf_hub_download(repo, filename, token=self.token, local_files_only=True))
            self._need_network(f"{repo}/{filename}")
            return Path(hf_hub_download(repo, filename, token=self.token))
        target.parent.mkdir(parents=True, exist_ok=True)
        log_message(f"Downloading {target.name} from Hugging Face ({repo})...", always_print=True)
        downloaded = Path(hf_hub_download(repo, filename, local_dir=str(target.parent), token=self.token))
        if downloaded != target:  # the repo keeps it in a subfolder / under another name
            try:
                downloaded.replace(target)
            except OSError:
                shutil.copyfile(downloaded, target)
                downloaded.unlink(missing_ok=True)
            _remove_empty_dirs(target.parent)
        log_message(f"Downloaded {target.name} successfully.", always_print=True)
        return target

    def hf_repo(
        self, repo: str, target: Path, revision: str | None = None, files: tuple[str, ...] | None = None
    ) -> Path:
        """A repository snapshot (only `files`, when given). With a commit `revision`, local
        files from another revision count as missing."""
        if not self.hf_cache and _complete(target, revision, files):
            return target
        if not self.hf_cache:
            self._need_network(repo)
        from huggingface_hub import snapshot_download

        if self.hf_cache:
            patterns = list(files) if files else None
            kwargs: dict[str, Any] = dict(revision=revision, token=self.token, allow_patterns=patterns)
            with contextlib.suppress(Exception):
                return Path(snapshot_download(repo, local_files_only=True, **kwargs))
            self._need_network(repo)
            return Path(snapshot_download(repo, **kwargs))
        target.mkdir(parents=True, exist_ok=True)
        log_message(f"Downloading repository {repo}...", always_print=True)
        try:
            snapshot_download(
                repo,
                local_dir=str(target),
                token=self.token,
                revision=revision,
                allow_patterns=list(files) if files else None,
            )
        except Exception as e:
            if target.resolve().is_relative_to(self.models_dir):
                shutil.rmtree(target, ignore_errors=True)
            raise ModelError(f"Failed to download repository {repo}: {e}") from e
        log_message(f"Downloaded repository {repo} successfully.", always_print=True)
        return target

    def url(self, url: str, target: Path, sha256: str) -> Path:
        import hashlib

        if not target.exists():
            self._need_network(target.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            log_message(f"Downloading {target.name}...", always_print=True)
            try:
                urllib.request.urlretrieve(url, part)
                part.replace(target)
            finally:
                part.unlink(missing_ok=True)
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
            raise ModelError(f"checksum mismatch for {target}")
        return target


def _remove_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        with contextlib.suppress(OSError):
            path.rmdir()


def _complete(target: Path, revision: str | None, files: tuple[str, ...] | None) -> bool:
    if files is None:  # a transformers checkpoint
        return target.is_dir() and any(
            (target / f).exists() for f in ("model.safetensors", "pytorch_model.bin")
        )
    if not all((target / f).exists() for f in files):
        return False
    if revision is None or not _COMMIT.fullmatch(revision):
        return True
    meta = target / ".cache" / "huggingface" / "download"
    for f in files:
        m = meta / f"{f}.metadata"
        if not m.exists() or m.read_text().split("\n", 1)[0].strip() != revision:
            return False
    return True


# --------------------------------------------------------------------------- loaders


@dataclass(frozen=True)
class LoadContext:
    device: Any  # torch.device
    dtype: Any  # torch.dtype
    fetch: Fetcher


def _state_dict(path: Path, device) -> dict:
    from safetensors.torch import load_file

    return load_file(str(path), device=str(device))


def _upscale(name: str) -> Callable[[LoadContext], Any]:
    def load(ctx: LoadContext):
        from spandrel import ModelLoader

        path = ctx.fetch.hf_file("Kim2091/2x-AnimeSharpV4", name, ctx.fetch.models_dir / "upscale" / name)
        return ModelLoader().load_from_state_dict(_state_dict(path, ctx.device)).to(ctx.device).eval()

    return load


def _yolo(repo: str, filename: str, model: Model) -> Callable[[LoadContext], Any]:
    def load(ctx: LoadContext):
        from ultralytics import YOLO

        return YOLO(str(ctx.fetch.hf_file(repo, filename, path_of(ctx.fetch.models_dir, model))))

    return load


def _conjoined(ctx: LoadContext):
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    from mangatl.ml.rtdetr_adapter import RTDetrYOLOAdapter

    path = ctx.fetch.hf_repo(
        "ogkalu/comic-text-and-bubble-detector", path_of(ctx.fetch.models_dir, Model.CONJOINED)
    )
    processor = RTDetrImageProcessor.from_pretrained(str(path))
    model = RTDetrV2ForObjectDetection.from_pretrained(str(path)).to(ctx.device).eval()
    return RTDetrYOLOAdapter(
        model=model, processor=processor, device=ctx.device, names=getattr(model.config, "id2label", None)
    )


def _manga_ocr(ctx: LoadContext):
    with contextlib.suppress(ImportError):  # MeCab (fugashi) needs to find its dictionary
        import unidic_lite

        os.environ["MECABRC"] = os.path.join(unidic_lite.DICDIR, "mecabrc")
    from manga_ocr import MangaOcr
    from transformers import logging as transformers_logging

    path = ctx.fetch.hf_repo(
        "kha-white/manga-ocr-base", path_of(ctx.fetch.models_dir, Model.MANGA_OCR), revision="refs/pr/4"
    )
    previous = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()  # position_ids mismatch warnings (transformers v5)
    try:
        return MangaOcr(pretrained_model_name_or_path=str(path), force_cpu=ctx.device.type == "cpu")
    finally:
        transformers_logging.set_verbosity(previous)


def _paddle_ocr_vl(ctx: LoadContext):
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from transformers import logging as transformers_logging

    path = str(
        ctx.fetch.hf_repo("PaddlePaddle/PaddleOCR-VL-1.6", path_of(ctx.fetch.models_dir, Model.PADDLE_OCR_VL))
    )
    previous = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()  # upstream config, RoPE and processor deprecations
    try:
        config = AutoConfig.from_pretrained(path)
        config.tie_word_embeddings = False
        if hasattr(config, "text_config"):
            config.text_config.tie_word_embeddings = False
        processor = AutoProcessor.from_pretrained(path, backend="torchvision")
        dtype = ctx.dtype if ctx.device.type == "cuda" else None
        for attn in ("flash_attention_2", "sdpa", "eager"):
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    path, config=config, dtype=dtype, attn_implementation=attn
                )
                return processor, model.to(ctx.device).eval()
            except (ImportError, ValueError, RuntimeError):
                if attn == "eager":
                    raise
        raise AssertionError("unreachable")
    finally:
        transformers_logging.set_verbosity(previous)


def _hayai(ctx: LoadContext):
    from mangatl.ocr.hayai import HayaiOCR

    base = ctx.fetch.models_dir / "ocr"
    weights = ctx.fetch.hf_repo(
        "JustANormalTinkerer/hayai-ocr-v2.5-nova",
        base / "hayai-ocr-v2.5-nova",
        revision=HAYAI_REVISION,
        files=("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"),
    )
    siglip = ctx.fetch.hf_repo(
        "google/siglip2-base-patch16-naflex",
        base / "siglip2-naflex",
        files=("config.json", "preprocessor_config.json"),
    )
    return HayaiOCR(weights, siglip, str(ctx.device))


def _text_lines(ctx: LoadContext):
    from mangatl.detect.text_lines import LineDetector

    return LineDetector(ctx.fetch.url(CTD_URL, path_of(ctx.fetch.models_dir, Model.TEXT_LINES), CTD_SHA256))


def _lama(ctx: LoadContext):
    from spandrel import ModelLoader

    path = ctx.fetch.hf_file(
        "mayocream/lama-manga", "lama-manga.safetensors", path_of(ctx.fetch.models_dir, Model.LAMA)
    )
    # the checkpoint dropped the "generator." prefix spandrel recognises LaMa by
    state = {f"generator.{k}": v for k, v in _state_dict(path, "cpu").items()}
    return ModelLoader().load_from_state_dict(state).to(ctx.device).eval()


def _text_sam(ctx: LoadContext):
    from mangatl.ml.hi_sam import build_text_sam_ts_l, to_inference_device

    path = ctx.fetch.hf_file(
        "mayocream/koharu-text-sam-ts-l", "model.safetensors", path_of(ctx.fetch.models_dir, Model.TEXT_SAM)
    )
    model = build_text_sam_ts_l()
    model.load_state_dict(_state_dict(path, "cpu"), strict=True)
    return to_inference_device(model, ctx.device).eval()


@dataclass(frozen=True)
class Spec:
    path: str  # weights, relative to models_dir
    load: Callable[[LoadContext], Any]
    weights_mb: int  # VRAM estimates, replaced by measurements once loaded / used
    work_mb: int  # memory a call needs beyond the weights
    keep: int  # under memory pressure the lowest goes first (then the least recently used)
    gpu: bool = True  # False: runs on the CPU (needs neither the GPU slot nor VRAM)


_Y = "detection/yolo"
SPECS: dict[Model, Spec] = {
    Model.UPSCALE: Spec(
        "upscale/2x-AnimeSharpV4_RCAN.safetensors", _upscale("2x-AnimeSharpV4_RCAN.safetensors"), 60, 600, 0
    ),
    Model.UPSCALE_LITE: Spec(
        "upscale/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
        _upscale("2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"),
        20,
        400,
        0,
    ),
    Model.BUBBLES: Spec(
        f"{_Y}/yolov8m_seg-speech-bubble.pt",
        _yolo("kitsumed/yolov8m_seg-speech-bubble", "model.pt", Model.BUBBLES),
        60,
        500,
        5,
    ),
    Model.BUBBLES_2: Spec(
        f"{_Y}/manga109-segmentation-bubble.pt",
        _yolo("huyvux3005/manga109-segmentation-bubble", "best.pt", Model.BUBBLES_2),
        120,
        900,
        5,
    ),
    Model.CONJOINED: Spec("detection/rtdetr/comic-text-and-bubble-detector", _conjoined, 180, 500, 5),
    Model.OUTSIDE_TEXT: Spec(
        f"{_Y}/animetext_yolov12x.pt",
        _yolo("deepghs/AnimeText_yolo", "yolo12x_animetext/model.pt", Model.OUTSIDE_TEXT),
        250,
        600,
        5,
    ),
    Model.PANELS: Spec(
        f"{_Y}/manga109_v2023.12.07_l_yolov11.pt",
        _yolo("deepghs/manga109_yolo", "v2023.12.07_l_yv11/model.pt", Model.PANELS),
        110,
        500,
        5,
    ),
    Model.MANGA_OCR: Spec("ocr/manga-ocr-base", _manga_ocr, 450, 200, 4),
    Model.PADDLE_OCR_VL: Spec("ocr/paddleocr-vl-1.6", _paddle_ocr_vl, 2000, 1500, 1),
    Model.HAYAI: Spec("ocr/hayai-ocr-v2.5-nova", _hayai, 650, 400, 4),
    Model.TEXT_LINES: Spec("detection/ctd/comictextdetector.pt.onnx", _text_lines, 0, 0, 9, gpu=False),
    Model.LAMA: Spec("inpainting/lama/lama-manga.safetensors", _lama, 200, 1500, 2),
    Model.TEXT_SAM: Spec("detection/text_sam/koharu-text-sam-ts-l.safetensors", _text_sam, 700, 1500, 3),
}


def path_of(models_dir: Path, model: Model) -> Path:
    """Where a model's weights live (its identity in cache keys)."""
    return Path(models_dir).resolve() / SPECS[model].path
