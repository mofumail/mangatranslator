"""Hayai OCR v2.5 Nova: crop-level manga OCR (JA/ZH/KO/EN) that keeps the crop's aspect ratio.

The model code is vendored (vendor/hayai, audited, pinned revision) and imported locally, so
no `trust_remote_code` is used; only weights, tokenizer and JSON configs are downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image


class HayaiOCR:
    def __init__(self, weights: Path, siglip: Path, device: str = "cuda"):
        """weights: the model's snapshot (mangatl.engine.models fetches it at the pinned
        revision); siglip: the SigLIP2 configs its vision tower and processor use."""
        from transformers import PreTrainedTokenizerFast, Siglip2ImageProcessor

        from vendor.hayai.configuration_hayai import HayaiConfig
        from vendor.hayai.modeling_hayai import HayaiModel

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
        inp = self.processor(
            images=[im.convert("RGB") for im in images], max_num_patches=max_patches, return_tensors="pt"
        ).to(self.device)
        # Plain PyTorch path: the fused engine records CUDA graphs, which crashed the GPU context
        # ("illegal memory access") while another thread inpainted (calls are serialised by the
        # engine's GPU slot now, but graphs would also pin memory between calls).
        return self.model.generate(
            pixel_values=inp["pixel_values"],
            pixel_attention_mask=inp["pixel_attention_mask"],
            spatial_shapes=inp["spatial_shapes"],
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
            use_cuda_graph=False,
            fast_engine=False,
        )
