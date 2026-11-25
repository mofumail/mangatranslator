import contextlib
import os
import shutil
import threading
from enum import Enum
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from spandrel import ModelLoader
from transformers import (
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
)
from ultralytics import YOLO

from mangatl.common.device import empty_cache, get_best_device, get_best_dtype, get_device_info
from mangatl.ml.rtdetr_adapter import RTDetrYOLOAdapter
from mangatl.common.download import download_file
from mangatl.common.exceptions import ModelError
from mangatl.common.log import log_message


class ModelType(Enum):
    """Enumeration of available model types."""

    UPSCALE = "upscale"
    UPSCALE_LITE = "upscale_lite"
    YOLO_SPEECH_BUBBLE = "yolo_speech_bubble"
    YOLO_SPEECH_BUBBLE_2 = "yolo_speech_bubble_2"
    RTDETR_CONJOINED_BUBBLE = "rtdetr_conjoined_bubble"
    YOLO_OSBTEXT = "yolo_osbtext"
    YOLO_PANEL = "yolo_panel"
    MANGA_OCR = "manga_ocr"
    PADDLE_OCR_VL = "paddle_ocr_vl"
    LAMA_MANGA = "lama_manga"
    TEXT_SAM = "text_sam"


# Old flat folder -> its place in the task-grouped layout
_LEGACY_MODEL_DIRS = (
    ("lama", "inpainting/lama"),
    ("yolo", "detection/yolo"),
    ("rtdetr", "detection/rtdetr"),
    ("manga-ocr-base", "ocr/manga-ocr-base"),
    ("paddleocr-vl-1.6", "ocr/paddleocr-vl-1.6"),
)


def migrate_model_dirs(models_dir: Path | str = "./models") -> None:
    """Move downloads from the old flat models/ layout into the task-grouped one.

    A folder is left where it is when its new location already exists.
    """
    models_dir = Path(models_dir)
    for old, new in _LEGACY_MODEL_DIRS:
        src, dst = models_dir / old, models_dir / new
        if not src.is_dir() or dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            src.rename(dst)
        except OSError as e:
            log_message(
                f"Could not move {src} to {dst} ({e}). Move it by hand to avoid "
                "downloading its models again.",
                always_print=True,
            )
            continue
        log_message(f"Moved {src} to {dst}", always_print=True)


class ModelManager:
    """Singleton model manager for MangaTranslator."""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the model manager (only once due to singleton pattern)."""
        with self._lock:
            if self._initialized:
                return

            self.device = get_best_device()
            self.dtype = get_best_dtype(self.device)

            self.models = {}
            self._apply_models_dir(Path("./models"))
            self.model_urls = self._init_model_urls()
            self.model_hf_repos = self._init_hf_repos()

            self.use_hf_cache = False
            self.hf_token = None
            # Serializes inpainting-model inference across threads
            self.flux_inference_lock = threading.Lock()
            # SAM-TS-L sits in RAM after park_text_sam until load_text_sam returns it
            self._text_sam_parked = False

            self._initialized = True
            log_message(
                f"Model Manager initialized on device: {self.device}", always_print=True
            )

    def _apply_models_dir(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir).resolve()
        self.model_paths = self._init_model_paths()


    def _init_model_paths(self):
        """Initialize model file paths."""
        model_dir = self.models_dir
        detection_dir = model_dir / "detection"
        return {
            ModelType.UPSCALE: (
                model_dir / "upscale" / "2x-AnimeSharpV4_RCAN.safetensors"
            ),
            ModelType.UPSCALE_LITE: (
                model_dir / "upscale" / "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"
            ),
            ModelType.YOLO_SPEECH_BUBBLE: (
                detection_dir / "yolo" / "yolov8m_seg-speech-bubble.pt"
            ),
            ModelType.YOLO_SPEECH_BUBBLE_2: (
                detection_dir / "yolo" / "manga109-segmentation-bubble.pt"
            ),
            ModelType.RTDETR_CONJOINED_BUBBLE: (
                detection_dir / "rtdetr" / "comic-text-and-bubble-detector"
            ),
            ModelType.YOLO_OSBTEXT: (detection_dir / "yolo" / "animetext_yolov12x.pt"),
            ModelType.YOLO_PANEL: (
                detection_dir / "yolo" / "manga109_v2023.12.07_l_yolov11.pt"
            ),
            ModelType.LAMA_MANGA: (
                model_dir / "inpainting" / "lama" / "lama-manga.safetensors"
            ),
            ModelType.TEXT_SAM: (
                detection_dir / "text_sam" / "koharu-text-sam-ts-l.safetensors"
            ),
            ModelType.MANGA_OCR: (model_dir / "ocr" / "manga-ocr-base"),
            ModelType.PADDLE_OCR_VL: (model_dir / "ocr" / "paddleocr-vl-1.6"),
        }

    def _init_model_urls(self):
        """Initialize model download URLs."""
        return {
            ModelType.UPSCALE: (
                "https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/"
                "2x-AnimeSharpV4_RCAN.safetensors"
            ),
            ModelType.UPSCALE_LITE: (
                "https://huggingface.co/Kim2091/2x-AnimeSharpV4/resolve/main/"
                "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"
            ),
        }

    def _init_hf_repos(self):
        """Initialize Hugging Face repository information."""
        repos = {
            ModelType.UPSCALE: {
                "repo_id": "Kim2091/2x-AnimeSharpV4",
                "filename": "2x-AnimeSharpV4_RCAN.safetensors",
            },
            ModelType.UPSCALE_LITE: {
                "repo_id": "Kim2091/2x-AnimeSharpV4",
                "filename": "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",
            },
            ModelType.YOLO_SPEECH_BUBBLE: {
                "repo_id": "kitsumed/yolov8m_seg-speech-bubble",
                "filename": "model.pt",
            },
            ModelType.YOLO_SPEECH_BUBBLE_2: {
                "repo_id": "huyvux3005/manga109-segmentation-bubble",
                "filename": "best.pt",
            },
            ModelType.RTDETR_CONJOINED_BUBBLE: {
                "repo_id": "ogkalu/comic-text-and-bubble-detector",
            },
            ModelType.YOLO_OSBTEXT: {
                "repo_id": "deepghs/AnimeText_yolo",
                "filename": "yolo12x_animetext/model.pt",
            },
            ModelType.YOLO_PANEL: {
                "repo_id": "deepghs/manga109_yolo",
                "filename": "v2023.12.07_l_yv11/model.pt",
            },
        }

        repos[ModelType.MANGA_OCR] = {
            "repo_id": "kha-white/manga-ocr-base",
            "revision": "refs/pr/4",
        }
        repos[ModelType.PADDLE_OCR_VL] = {
            "repo_id": "PaddlePaddle/PaddleOCR-VL-1.6",
        }
        repos[ModelType.LAMA_MANGA] = {
            "repo_id": "mayocream/lama-manga",
            "filename": "lama-manga.safetensors",
        }
        repos[ModelType.TEXT_SAM] = {
            "repo_id": "mayocream/koharu-text-sam-ts-l",
            "filename": "model.safetensors",
        }

        return repos

    def _enable_hf_xet_high_performance(self) -> None:
        if "HF_XET_HIGH_PERFORMANCE" not in os.environ:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    def _hf_cache_dir(self, local_dir: Path | str) -> str | None:
        """cache_dir for from_pretrained; None selects the default HF cache."""
        return None if self.use_hf_cache else str(local_dir)

    def _remove_empty_child_dirs(self, root: Path, verbose: bool = False) -> None:
        if not root.exists():
            return
        for path in sorted(
            (child for child in root.rglob("*") if child.is_dir()),
            key=lambda child: len(child.parts),
            reverse=True,
        ):
            try:
                path.rmdir()
                log_message(
                    f"Removed empty download directory {path}.", verbose=verbose
                )
            except OSError:
                pass


    def _ensure_hf_file(
        self,
        repo_id: str,
        filename: str,
        target: Path,
        token: str | None = None,
        verbose: bool = False,
    ) -> Path:
        """Download file from Hugging Face if it doesn't exist.

        Args:
            repo_id: Hugging Face repository ID
            filename: Name of file to download
            target: Path where file should be saved
            token: Optional Hugging Face token
            verbose: Whether to print verbose logging

        Returns:
            Path to the file (inside the HF cache when use_hf_cache is set)
        """
        if self.use_hf_cache:
            self._enable_hf_xet_high_performance()
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=token or self.hf_token or os.environ.get("HF_TOKEN"),
                )
            )
        if target.exists():
            self._remove_empty_child_dirs(target.parent, verbose=verbose)
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        log_message(
            f"Downloading {target.name} from Hugging Face ({repo_id})...",
            verbose=verbose,
        )
        self._enable_hf_xet_high_performance()
        effective_token = token or self.hf_token or os.environ.get("HF_TOKEN")
        if effective_token:
            os.environ["HF_TOKEN"] = effective_token
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target.parent),
            token=effective_token,
        )
        downloaded_path = Path(downloaded)
        if downloaded_path != target:
            try:
                downloaded_path.replace(target)
            except Exception:
                shutil.copyfile(downloaded_path, target)
                with contextlib.suppress(Exception):
                    downloaded_path.unlink()

            self._remove_empty_child_dirs(target.parent, verbose=verbose)
        log_message(f"Downloaded {target.name} successfully.", verbose=verbose)
        return target

    def _ensure_hf_repo(
        self,
        repo_id: str,
        target_dir: Path,
        token: str | None = None,
        revision: str | None = None,
        verbose: bool = False,
    ) -> Path:
        """Download entire repository from Hugging Face if it doesn't exist.

        Args:
            repo_id: Hugging Face repository ID
            target_dir: Directory where repository should be saved
            token: Optional Hugging Face token
            revision: Optional git revision (branch, tag, commit hash)
            verbose: Whether to print verbose logging

        Returns:
            Path to the downloaded repository directory
        """
        if self.use_hf_cache:
            self._enable_hf_xet_high_performance()
            return Path(
                snapshot_download(
                    repo_id=repo_id,
                    token=token or self.hf_token or os.environ.get("HF_TOKEN"),
                    revision=revision,
                )
            )
        # Check for safetensors for transformers v5+
        safetensors_file = target_dir / "model.safetensors"
        bin_file = target_dir / "pytorch_model.bin"

        is_downloaded = False
        if target_dir.exists() and (
            safetensors_file.exists() or (bin_file.exists() and revision is None)
        ):
            is_downloaded = True

        if is_downloaded:
            return target_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        log_message(
            (
                f"Downloading repository {repo_id} (revision: {revision})..."
                if revision
                else f"Downloading repository {repo_id}..."
            ),
            verbose=verbose,
        )
        self._enable_hf_xet_high_performance()
        effective_token = token or self.hf_token or os.environ.get("HF_TOKEN")
        if effective_token:
            os.environ["HF_TOKEN"] = effective_token
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                token=effective_token,
                revision=revision,
            )
            log_message(
                f"Downloaded repository {repo_id} successfully.", verbose=verbose
            )
        except Exception as e:
            if target_dir.exists():
                target_resolved = target_dir.resolve()
                try:
                    target_resolved.relative_to(self.models_dir)
                    with contextlib.suppress(Exception):
                        shutil.rmtree(target_dir)
                except ValueError:
                    # target_dir is not within models/, skip deletion for safety
                    log_message(
                        f"Warning: Skipping deletion of {target_dir} as it is outside models/ directory",
                        always_print=True,
                    )
            raise ModelError(f"Failed to download repository {repo_id}: {e}") from e
        return target_dir

    def is_loaded(self, model_type: ModelType) -> bool:
        """Check if a model is currently loaded."""
        with self._lock:
            return model_type in self.models and self.models[model_type] is not None

    def load_upscale(self, verbose: bool = False):
        """Load upscale model (AnimeSharpV4 RCAN)."""
        with self._lock:
            if self.is_loaded(ModelType.UPSCALE):
                return self.models[ModelType.UPSCALE]

            log_message(
                "Loading upscale model (2x-AnimeSharpV4_RCAN)...", verbose=verbose
            )
            path = self.model_paths[ModelType.UPSCALE]

            # Try HF download first, fallback to direct URL
            try:
                hf_info = self.model_hf_repos[ModelType.UPSCALE]
                path = self._ensure_hf_file(
                    hf_info["repo_id"], hf_info["filename"], path, verbose=verbose
                )
            except Exception:
                download_file(path, self.model_urls[ModelType.UPSCALE])

            if path.suffix == ".safetensors":
                from safetensors import safe_open

                state_dict = {}
                with safe_open(path, framework="pt", device=str(self.device)) as f:
                    for key in f.keys():  # noqa: SIM118
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(
                    path, map_location=self.device, weights_only=False
                )

            model = (
                ModelLoader().load_from_state_dict(state_dict).to(self.device).eval()
            )
            self.models[ModelType.UPSCALE] = model
            log_message("Upscale model loaded.", verbose=verbose)
            return model

    def load_upscale_lite(self, verbose: bool = False):
        """Load upscale lite model (AnimeSharpV4 Fast RCAN PU)."""
        with self._lock:
            if self.is_loaded(ModelType.UPSCALE_LITE):
                return self.models[ModelType.UPSCALE_LITE]

            log_message(
                "Loading upscale lite model (2x-AnimeSharpV4_Fast_RCAN_PU)...",
                verbose=verbose,
            )
            path = self.model_paths[ModelType.UPSCALE_LITE]

            # Try HF download first, fallback to direct URL
            try:
                hf_info = self.model_hf_repos[ModelType.UPSCALE_LITE]
                path = self._ensure_hf_file(
                    hf_info["repo_id"], hf_info["filename"], path, verbose=verbose
                )
            except Exception:
                download_file(path, self.model_urls[ModelType.UPSCALE_LITE])

            if path.suffix == ".safetensors":
                from safetensors import safe_open

                state_dict = {}
                with safe_open(path, framework="pt", device=str(self.device)) as f:
                    for key in f.keys():  # noqa: SIM118
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(
                    path, map_location=self.device, weights_only=False
                )

            model = (
                ModelLoader().load_from_state_dict(state_dict).to(self.device).eval()
            )
            self.models[ModelType.UPSCALE_LITE] = model
            log_message("Upscale lite model loaded.", verbose=verbose)
            return model

    def _resolve_speech_bubble_model_type(self, model_path: str | None) -> ModelType:
        """Determine which speech bubble model type a path corresponds to."""
        if model_path is None:
            return ModelType.YOLO_SPEECH_BUBBLE
        # Callers may pass a path relative to the working directory
        p = Path(model_path).resolve()
        if p == self.model_paths[ModelType.YOLO_SPEECH_BUBBLE_2]:
            return ModelType.YOLO_SPEECH_BUBBLE_2
        return ModelType.YOLO_SPEECH_BUBBLE

    def load_yolo_speech_bubble(
        self, model_path: str | None = None, verbose: bool = False
    ):
        """Load YOLO model for speech bubble detection.

        Args:
            model_path: Optional custom path to YOLO model. If None, uses default.
            verbose: Whether to print verbose logging
        """
        with self._lock:
            model_type = self._resolve_speech_bubble_model_type(model_path)

            if self.is_loaded(model_type):
                return self.models[model_type]

            log_message(
                "Loading YOLO speech bubble detection model...", verbose=verbose
            )

            path = (
                self.model_paths[model_type]
                if model_path is None
                else Path(model_path).resolve()
            )

            if path == self.model_paths[model_type]:
                hf_info = self.model_hf_repos[model_type]
                path = self._ensure_hf_file(
                    hf_info["repo_id"], hf_info["filename"], path, verbose=verbose
                )

            model = YOLO(str(path))
            self.models[model_type] = model
            log_message("YOLO model loaded.", verbose=verbose)
            return model

    def load_rtdetr_conjoined_bubble(self, verbose: bool = False):
        """Load RT-DETR-v2 model for conjoined/fallback bubble detection.

        Returns a YOLO-compatible adapter exposing ``.names`` and a callable
        predict API used by the detection pipeline.
        """
        with self._lock:
            if self.is_loaded(ModelType.RTDETR_CONJOINED_BUBBLE):
                return self.models[ModelType.RTDETR_CONJOINED_BUBBLE]

            log_message(
                "Loading RT-DETR conjoined bubble detection model...", verbose=verbose
            )
            path = self.model_paths[ModelType.RTDETR_CONJOINED_BUBBLE]
            hf_info = self.model_hf_repos[ModelType.RTDETR_CONJOINED_BUBBLE]
            path = self._ensure_hf_repo(hf_info["repo_id"], path, verbose=verbose)

            try:
                processor = RTDetrImageProcessor.from_pretrained(str(path))
                model = RTDetrV2ForObjectDetection.from_pretrained(str(path))
                model.to(self.device)
                model.eval()
                adapter = RTDetrYOLOAdapter(
                    model=model,
                    processor=processor,
                    device=self.device,
                    names=getattr(model.config, "id2label", None),
                )
            except Exception as e:
                raise ModelError(f"Failed to load RT-DETR conjoined model: {e}") from e

            self.models[ModelType.RTDETR_CONJOINED_BUBBLE] = adapter
            log_message("RT-DETR conjoined bubble model loaded.", verbose=verbose)
            return adapter

    def load_yolo_osbtext(self, token: str | None = None, verbose: bool = False):
        """Load YOLO model for outside text detection.

        Args:
            token: Hugging Face token for gated repo access.
            verbose: Whether to print verbose logging
        """
        with self._lock:
            if self.is_loaded(ModelType.YOLO_OSBTEXT):
                return self.models[ModelType.YOLO_OSBTEXT]

            log_message("Loading YOLO OSB Text detection model...", verbose=verbose)

            path = self.model_paths[ModelType.YOLO_OSBTEXT]
            hf_info = self.model_hf_repos[ModelType.YOLO_OSBTEXT]

            path = self._ensure_hf_file(
                hf_info["repo_id"],
                hf_info["filename"],
                path,
                token=token,
                verbose=verbose,
            )

            model = YOLO(str(path))
            self.models[ModelType.YOLO_OSBTEXT] = model
            log_message("YOLO OSB Text model loaded.", verbose=verbose)
            return model

    def load_yolo_panel(self, verbose: bool = False):
        """Load YOLO model for panel detection.

        Args:
            verbose: Whether to print verbose logging
        """
        with self._lock:
            if self.is_loaded(ModelType.YOLO_PANEL):
                return self.models[ModelType.YOLO_PANEL]

            log_message("Loading YOLO panel detection model...", verbose=verbose)
            path = self.model_paths[ModelType.YOLO_PANEL]
            hf_info = self.model_hf_repos[ModelType.YOLO_PANEL]

            path = self._ensure_hf_file(
                hf_info["repo_id"],
                hf_info["filename"],
                path,
                verbose=verbose,
            )

            model = YOLO(str(path))
            self.models[ModelType.YOLO_PANEL] = model
            log_message("YOLO panel model loaded.", verbose=verbose)
            return model

    def load_manga_ocr(self, verbose: bool = False) -> Path:
        """Ensure manga-ocr model repository is downloaded.

        Args:
            verbose: Whether to print verbose logging

        Returns:
            Path to the downloaded manga-ocr model directory
        """
        with self._lock:
            model_path = self.model_paths[ModelType.MANGA_OCR]
            hf_info = self.model_hf_repos[ModelType.MANGA_OCR]
            model_path = self._ensure_hf_repo(
                hf_info["repo_id"],
                model_path,
                revision=hf_info.get("revision"),
                verbose=verbose,
            )
            log_message("manga-ocr model repository ready.", verbose=verbose)
            return model_path

    def get_manga_ocr(self, verbose: bool = False):
        """Get manga-ocr instance, loading it if necessary.

        Args:
            verbose: Whether to print verbose logging

        Returns:
            MangaOcr instance
        """
        with self._lock:
            if self.is_loaded(ModelType.MANGA_OCR):
                return self.models[ModelType.MANGA_OCR]

            log_message("Initializing manga-ocr...", verbose=verbose)

            # Fix for MeCab/Fugashi on non-Windows systems
            try:
                import os

                import unidic_lite

                os.environ["MECABRC"] = os.path.join(unidic_lite.DICDIR, "mecabrc")
            except ImportError:
                log_message(
                    "Warning: unidic_lite not found, skipping MeCab fix",
                    verbose=verbose,
                )
            except Exception as e:
                log_message(f"Warning: Failed to apply MeCab fix: {e}", verbose=verbose)

            from manga_ocr import MangaOcr
            from transformers import logging as transformers_logging

            model_path = self.load_manga_ocr(verbose=verbose)

            # Suppress loading warnings (specifically for position_ids mismatch in transformers v5)
            previous_level = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()
            try:
                manga_ocr_instance = MangaOcr(
                    pretrained_model_name_or_path=str(model_path),
                    force_cpu=self.device.type == "cpu",
                )
            finally:
                transformers_logging.set_verbosity(previous_level)

            self.models[ModelType.MANGA_OCR] = manga_ocr_instance
            log_message("manga-ocr initialized", verbose=verbose)
            return manga_ocr_instance

    def load_paddle_ocr_vl(self, verbose: bool = False) -> Path:
        """Ensure PaddleOCR-VL-1.6 model repository is downloaded.

        Args:
            verbose: Whether to print verbose logging

        Returns:
            Path to the downloaded PaddleOCR-VL-1.6 model directory
        """
        with self._lock:
            model_path = self.model_paths[ModelType.PADDLE_OCR_VL]
            hf_info = self.model_hf_repos[ModelType.PADDLE_OCR_VL]
            model_path = self._ensure_hf_repo(
                hf_info["repo_id"],
                model_path,
                revision=hf_info.get("revision"),
                verbose=verbose,
            )
            log_message("PaddleOCR-VL-1.6 model repository ready.", verbose=verbose)
            return model_path

    def get_paddle_ocr_vl(self, verbose: bool = False):
        """Get PaddleOCR-VL-1.6 (processor, model) tuple, loading if necessary.

        Args:
            verbose: Whether to print verbose logging

        Returns:
            Tuple of (AutoProcessor, AutoModelForImageTextToText)
        """
        with self._lock:
            if self.is_loaded(ModelType.PADDLE_OCR_VL):
                return self.models[ModelType.PADDLE_OCR_VL]

            log_message("Initializing PaddleOCR-VL-1.6...", verbose=verbose)

            from transformers import (
                AutoConfig,
                AutoModelForImageTextToText,
                AutoProcessor,
            )
            from transformers import logging as transformers_logging

            model_path = self.load_paddle_ocr_vl(verbose=verbose)
            model_path_str = str(model_path)
            token = self.hf_token or os.environ.get("HF_TOKEN")

            # Suppress upstream config, RoPE, and processor deprecation warnings from normal logs
            previous_level = None
            if not verbose:
                previous_level = transformers_logging.get_verbosity()
                transformers_logging.set_verbosity_error()
            try:
                config = AutoConfig.from_pretrained(model_path_str, token=token)
                config.tie_word_embeddings = False
                if hasattr(config, "text_config"):
                    config.text_config.tie_word_embeddings = False

                processor = AutoProcessor.from_pretrained(
                    model_path_str, token=token, backend="torchvision"
                )

                # Prefer flash_attention_2 on CUDA, fall back to sdpa on Windows/CPU
                dtype = self.dtype if self.device.type == "cuda" else None
                for attn_impl in ("flash_attention_2", "sdpa", "eager"):
                    try:
                        model = (
                            AutoModelForImageTextToText.from_pretrained(
                                model_path_str,
                                config=config,
                                dtype=dtype,
                                attn_implementation=attn_impl,
                                token=token,
                            )
                            .to(self.device)
                            .eval()
                        )
                        log_message(
                            f"PaddleOCR-VL-1.6 loaded with {attn_impl}", verbose=verbose
                        )
                        break
                    except (ImportError, ValueError, RuntimeError):
                        if attn_impl == "eager":
                            raise
                        log_message(
                            f"PaddleOCR-VL-1.6: {attn_impl} unavailable, trying fallback",
                            verbose=verbose,
                        )
            finally:
                if previous_level is not None:
                    transformers_logging.set_verbosity(previous_level)

            self.models[ModelType.PADDLE_OCR_VL] = (processor, model)
            log_message("PaddleOCR-VL-1.6 initialized.", verbose=verbose)
            return self.models[ModelType.PADDLE_OCR_VL]


    def set_device(self, device: torch.device):
        """Override the auto-detected device (e.g. --cpu). Call before loading models."""
        with self._lock:
            self.device = device
            self.dtype = get_best_dtype(device)

    def set_models_dir(self, models_dir: Path | str):
        """Store downloads under `models_dir` (e.g. --models). Call before loading models."""
        with self._lock:
            self._apply_models_dir(Path(models_dir))

    def set_hf_token(self, token: str, enable_fast_download: bool = True):
        """Set the global HuggingFace token for model downloads.

        Args:
            token: HuggingFace API token
            enable_fast_download: If True, enables HF_XET_HIGH_PERFORMANCE for faster downloads
        """
        with self._lock:
            self.hf_token = token if token else None
            if self.hf_token:
                os.environ["HF_TOKEN"] = self.hf_token
                if enable_fast_download:
                    self._enable_hf_xet_high_performance()
            elif "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]


    def load_lama_manga(self, verbose: bool = False):
        """Load the manga-finetuned Big-LaMa inpainting model through spandrel."""
        with self._lock:
            if self.is_loaded(ModelType.LAMA_MANGA):
                return self.models[ModelType.LAMA_MANGA]

            log_message("Loading LaMa manga inpainting model...", verbose=verbose)
            from safetensors.torch import load_file

            hf_info = self.model_hf_repos[ModelType.LAMA_MANGA]
            path = self._ensure_hf_file(
                hf_info["repo_id"],
                hf_info["filename"],
                self.model_paths[ModelType.LAMA_MANGA],
                verbose=verbose,
            )
            # The checkpoint dropped the "generator." prefix that spandrel detects LaMa by
            state_dict = {
                f"generator.{key}": tensor
                for key, tensor in load_file(str(path)).items()
            }
            model = (
                ModelLoader().load_from_state_dict(state_dict).to(self.device).eval()
            )
            self.models[ModelType.LAMA_MANGA] = model
            log_message("LaMa manga inpainting model loaded.", verbose=verbose)
            return model

    def load_text_sam(self, verbose: bool = False):
        """Load Koharu's manga-tuned SAM-TS-L, which segments text pixels on a page."""
        with self._lock:
            if self.is_loaded(ModelType.TEXT_SAM):
                # A no-op unless park_text_sam moved it to RAM
                self._text_sam_parked = False
                return self.models[ModelType.TEXT_SAM].to(self.device)

            log_message("Loading SAM-TS-L text segmentation model...", verbose=verbose)
            from safetensors.torch import load_file

            from mangatl.ml.hi_sam import build_text_sam_ts_l, to_inference_device

            hf_info = self.model_hf_repos[ModelType.TEXT_SAM]
            path = self._ensure_hf_file(
                hf_info["repo_id"],
                hf_info["filename"],
                self.model_paths[ModelType.TEXT_SAM],
                verbose=verbose,
            )
            model = build_text_sam_ts_l()
            model.load_state_dict(load_file(str(path)), strict=True)
            model = to_inference_device(model, self.device).eval()
            self.models[ModelType.TEXT_SAM] = model
            self._text_sam_parked = False
            log_message("SAM-TS-L text segmentation model loaded.", verbose=verbose)
            return model

    def unload_model(
        self, model_type: ModelType, force_gc: bool = True, verbose: bool = False
    ):
        """Unload a specific model and free memory.

        Args:
            model_type: Type of model to unload
            force_gc: Whether to force garbage collection
            verbose: Whether to print verbose logging
        """
        with self._lock:
            if not self.is_loaded(model_type):
                return

            log_message(f"Unloading {model_type.value}...", verbose=verbose)
            del self.models[model_type]
            self.models[model_type] = None

            if force_gc:
                empty_cache(self.device)

    def _unload_models(self, model_types, verbose: bool = False) -> bool:
        """Unload each loaded model in `model_types`; True if any was loaded."""
        loaded = [
            model_type for model_type in model_types if self.is_loaded(model_type)
        ]
        for model_type in loaded:
            self.unload_model(model_type, force_gc=False, verbose=verbose)
        if loaded:
            empty_cache(self.device)
        return bool(loaded)

    def unload_upscale_models(self, verbose: bool = False):
        """Unload upscale models (both regular and lite)."""
        self._unload_models((ModelType.UPSCALE, ModelType.UPSCALE_LITE), verbose)
        log_message("Upscale models unloaded.", verbose=verbose)

    def unload_ocr_models(self, verbose: bool = False):
        """Unload OCR-related models (YOLO and manga-ocr)."""
        ocr_models = (
            ModelType.YOLO_SPEECH_BUBBLE,
            ModelType.YOLO_SPEECH_BUBBLE_2,
            ModelType.RTDETR_CONJOINED_BUBBLE,
            ModelType.YOLO_OSBTEXT,
            ModelType.YOLO_PANEL,
            ModelType.MANGA_OCR,
            ModelType.PADDLE_OCR_VL,
        )
        if self._unload_models(ocr_models, verbose):
            log_message("OCR models unloaded.", verbose=verbose)


    def unload_lama_manga(self, verbose: bool = False):
        """Unload the LaMa manga inpainting model and the text segmenter it uses."""
        if self._unload_models((ModelType.LAMA_MANGA, ModelType.TEXT_SAM), verbose):
            log_message("LaMa manga inpainting model unloaded.", verbose=verbose)

    def park_text_sam(self, verbose: bool = False):
        """Free the SAM-TS-L text model's VRAM for a diffusion inpainter, keeping it in
        RAM (~0.7 GB) for the next page; a no-op once parked.

        Moving it back takes ~0.15 s on CUDA, against ~2.2 s to reload it from disk. On
        the CPU it is unloaded instead, since there it would stay in fp32 (~1.4 GB) while
        the diffusion inpainter needs the RAM.
        """
        if self.device.type == "cpu":
            if self._unload_models((ModelType.TEXT_SAM,), verbose):
                log_message(
                    "SAM-TS-L text segmentation model unloaded.", verbose=verbose
                )
            return
        with self._lock:
            if not self.is_loaded(ModelType.TEXT_SAM) or self._text_sam_parked:
                return
            self.models[ModelType.TEXT_SAM].to("cpu")
            self._text_sam_parked = True
            empty_cache(self.device)
        log_message("SAM-TS-L text segmentation model moved to RAM.", verbose=verbose)

    def unload_all(self, verbose: bool = False):
        """Unload all models and free all GPU memory."""
        log_message("Unloading all models...", verbose=verbose)
        with self._lock:
            for model_type in list(self.models.keys()):
                if self.is_loaded(model_type):
                    del self.models[model_type]
                    self.models[model_type] = None

            empty_cache(self.device)
            log_message("All models unloaded.", verbose=verbose)

    def get_memory_stats(self):
        """Get current GPU memory usage statistics."""
        return get_device_info(self.device)

    def print_memory_stats(self):
        """Print current GPU memory usage."""
        stats = self.get_memory_stats()
        if stats["memory"] == "N/A":
            log_message(f"Device: {stats['device']}", always_print=True)
        else:
            log_message(
                f"GPU Memory - Allocated: {stats['allocated_gb']} GB, "
                f"Reserved: {stats['reserved_gb']} GB",
                always_print=True,
            )

    def clear_cache(self):
        """Release unused GPU memory from PyTorch's CUDA cache."""
        empty_cache(self.device)


_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
