from dataclasses import dataclass, field
from functools import partial
from typing import Any

import cv2
import numpy as np
from PIL import Image

from mangatl.clean.glyph_mask import (
    UpscaledGlyphMap,
    glyph_dilation_radius,
    page_glyph_probability,
    refine_box_mask_to_glyphs,
    refine_mask_to_glyphs,
)
from mangatl.common.exceptions import CancellationError
from mangatl.common.log import log_message
from mangatl.engine import Model
from mangatl.ml.hi_sam import predict_text_probability
from mangatl.settings import Settings


@dataclass
class PageGlyphMap:
    """Text probability of the original page, computed on first use.

    Bubble cleaning, OSB tilt detection, LaMa's glyph masks, and the edit models' glyph
    composites share it, so the text segmentation model runs at most once per page, and
    not at all on pages that need none of them.
    """

    image: Image.Image
    models: Any  # the engine's models (mangatl.engine.Models)
    verbose: bool = False
    # LaMa masks add text found in each box at twice the text scale
    upscale_boxes: bool = False
    # Off by the user's choice; consumers then get no map, as when the model fails
    enabled: bool = True
    prob: np.ndarray | None = None
    failed: bool = False
    _upscaled: UpscaledGlyphMap | None = field(default=None, init=False, repr=False)

    @classmethod
    def for_page(cls, image: Image.Image, config: Settings, verbose: bool) -> "PageGlyphMap":
        return cls(
            image,
            config.models,
            verbose=verbose,
            upscale_boxes=config.cleaning.inpainter == "lama_manga",
            enabled=config.cleaning.text_segmentation,
        )

    def get(self) -> np.ndarray | None:
        """The probability map, or None when the text model can't be loaded or run."""
        if self.prob is None and self.enabled and not self.failed:
            try:
                # The GPU slot also makes this build once: another caller waits, then finds it
                with self.models.use(Model.TEXT_SAM) as sam:
                    if self.prob is None and not self.failed:
                        self.prob = page_glyph_probability(self.image, partial(predict_text_probability, sam))
            except CancellationError:
                raise
            except Exception as e:
                self.failed = True
                log_message(
                    f"Text segmentation unavailable ({e}); cleaning and inpainting "
                    "without glyph masks, and using heuristic tilt",
                    always_print=True,
                )
        return self.prob

    def box_probability(self, box: tuple[int, int, int, int]) -> np.ndarray | None:
        """Text probability inside `box` at twice the text scale, or None when that
        pass doesn't run for this page."""
        if not self.upscale_boxes or self.get() is None:
            return None
        try:
            with self.models.use(Model.TEXT_SAM) as sam:
                if self._upscaled is None:
                    self._upscaled = UpscaledGlyphMap(self.image)
                return self._upscaled.box_probability(box, partial(predict_text_probability, sam))
        except CancellationError:
            raise
        except Exception as e:
            self.upscale_boxes = False
            self._upscaled = None
            log_message(
                f"Upscaled text segmentation failed ({e}); LaMa masks use the page map alone",
                always_print=True,
            )
            return None

    def lama_mask(self, box_mask: np.ndarray) -> np.ndarray:
        """`box_mask` shrunk to its dilated glyphs, including those found inside its
        bounds at twice the text scale; `box_mask` itself without a map."""
        glyph_prob = self.get()
        if glyph_prob is None:
            return box_mask
        radius = glyph_dilation_radius(self.image.size)
        x, y, w, h = cv2.boundingRect(np.asarray(box_mask, np.uint8))
        box = (x, y, x + w, y + h)
        box_prob = self.box_probability(box) if w and h else None
        if box_prob is None:
            return refine_mask_to_glyphs(box_mask, glyph_prob, radius)
        return refine_box_mask_to_glyphs(box_mask, glyph_prob, box, box_prob, radius)
