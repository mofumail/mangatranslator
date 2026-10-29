"""Hayai OCR v2.5 Nova: crop-level manga OCR (JA/ZH/KO/EN) that keeps the crop's aspect ratio.

The model code is vendored (vendor/hayai, audited, pinned revision) and imported locally, so
no `trust_remote_code` is used; only weights, tokenizer and JSON configs are downloaded.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import torch
from PIL import Image

REPO = "JustANormalTinkerer/hayai-ocr-v2.5-nova"
REVISION = "2803d0dc005fa4396ab1e7c5a472d2c450a247f7"  # matches vendor/hayai
SIGLIP = "google/siglip2-base-patch16-naflex"

_lock = threading.Lock()
_instance: "HayaiOCR | None" = None


class HayaiOCR:
    def __init__(self, models_dir: Path, device: str = "cuda"):
        from huggingface_hub import snapshot_download
        from transformers import PreTrainedTokenizerFast, Siglip2ImageProcessor

        from vendor.hayai.configuration_hayai import HayaiConfig
        from vendor.hayai.modeling_hayai import HayaiModel

        base = Path(models_dir) / "ocr"
        weights = snapshot_download(
            REPO, revision=REVISION, local_dir=base / "hayai-ocr-v2.5-nova",
            allow_patterns=["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"])
        siglip = snapshot_download(SIGLIP, local_dir=base / "siglip2-naflex",
                                   allow_patterns=["config.json", "preprocessor_config.json"])

        cfg = HayaiConfig.from_pretrained(weights)
        cfg.vision_config = json.loads((Path(siglip) / "config.json").read_text())["vision_config"]
        self.device = device
        self.model = HayaiModel.from_pretrained(weights, config=cfg).to(device).eval()
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(weights)
        self.processor = Siglip2ImageProcessor.from_pretrained(siglip)

    @torch.inference_mode()
    def read(self, images: list[Image.Image], max_patches: int = 512, max_new_tokens: int = 256) -> list[str]:
        if not images:
            return []
        inp = self.processor(images=[im.convert("RGB") for im in images],
                             max_num_patches=max_patches, return_tensors="pt").to(self.device)
        return self.model.generate(
            pixel_values=inp["pixel_values"], pixel_attention_mask=inp["pixel_attention_mask"],
            spatial_shapes=inp["spatial_shapes"], tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens, repetition_penalty=1.0)


def get(models_dir: Path, device: str = "cuda") -> HayaiOCR:
    global _instance
    with _lock:
        if _instance is None:
            _instance = HayaiOCR(models_dir, device)
        return _instance
