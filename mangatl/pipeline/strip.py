"""Long strips (webtoons): split at text-free rows and translate section by section."""

import math
import tempfile
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from mangatl.common.cancellation import CancellationManager
from mangatl.common.image_utils import save_image_with_compression
from mangatl.common.log import log_message
from mangatl.detect.bubbles import detect_speech_bubbles
from mangatl.settings import Settings

# Section heights are multiples of the width: bubble detection recall peaks around 4-6x,
# dropping for whole strips (small bubbles lost) and page-shaped sections (large ones lost).
SPLIT_RATIO = 8.0
MIN_RATIO, TARGET_RATIO, MAX_RATIO = 3.0, 5.0, 7.0
MIN_GAP_RATIO = 0.045  # exceeds line spacing of bubble-less narration text
BLANK_TOLERANCE = 8


def _pick_gap(free: np.ndarray, min_gap: int, prefer: int) -> int | None:
    """Middle row of the run of free rows (>= min_gap long) nearest to `prefer`."""
    padded = np.concatenate(([False], free, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    middles = [
        int(start + end) // 2
        for start, end in zip(edges[::2], edges[1::2], strict=False)
        if end - start >= min_gap
    ]
    return min(middles, key=lambda row: abs(row - prefer), default=None)


def find_cut_rows(
    image: Image.Image,
    detect_rows: Callable[[Image.Image], list[tuple[int, int]]] | None = None,
) -> list[int]:
    """Rows at which to split a long strip into sections, or [] if it should stay whole.

    Cuts go through blank gutters when one is in range. Otherwise `detect_rows` (returning
    the (top, bottom) row ranges occupied by text in a crop) picks a row between text.
    """
    w, h = image.size
    if h <= SPLIT_RATIO * w:
        return []

    blank = np.ptp(np.asarray(image.convert("L")), axis=1) <= BLANK_TOLERANCE
    min_gap = max(8, int(w * MIN_GAP_RATIO))
    last = int(h - MIN_RATIO * w)
    cuts: list[int] = []
    pos = 0
    while h - pos > MAX_RATIO * w:
        cut = None
        lo = int(pos + MIN_RATIO * w)
        while cut is None and lo < last:
            hi = min(lo + int((MAX_RATIO - MIN_RATIO) * w), last)
            prefer = int(pos + TARGET_RATIO * w) - lo
            cut = _pick_gap(blank[lo:hi], min_gap, prefer)
            if cut is None and detect_rows is not None:
                free = np.ones(hi - lo, dtype=bool)
                for top, bottom in detect_rows(image.crop((0, lo, w, hi))):
                    free[max(0, int(top)) : max(0, int(bottom))] = False
                cut = _pick_gap(free, min_gap, prefer)
            if cut is not None:
                cut += lo
            lo = hi
        if cut is None:
            # Never cut through text: the rest of the strip stays one section.
            break
        cuts.append(cut)
        pos = cut
    return cuts


def detect_strip_text_rows(crop: Image.Image, image_path: Path, config: Settings) -> list[tuple[int, int]]:
    """Row ranges occupied by bubbles and text in a strip crop, for choosing cut rows."""
    bubbles, text_free_boxes = detect_speech_bubbles(
        image_path,
        config.yolo_model_path,
        config.detection.confidence,
        verbose=config.verbose,
        device=config.device,
        conjoined_detection=config.detection.conjoined,
        conjoined_confidence=config.detection.conjoined_confidence,
        image_override=crop,
        bubble_detector_model=config.detection.bubble_model,
    )
    boxes = [bubble["bbox"] for bubble in bubbles] + list(text_free_boxes)
    return [(int(box[1]), int(box[3])) for box in boxes]


def translate_strip(
    image_path: Path,
    strip: Image.Image,
    cut_rows: list[int],
    config: Settings,
    output_path: str | Path | None,
    target_mode: str,
    cancellation_manager: Optional["CancellationManager"],
    previous_context_texts: list[list[str]],
    previous_context_texts_provider: Callable[[], list[list[str]]] | None,
    ocr_texts_out: list[str] | None,
    empty_bubbles_out: list | None = None,
) -> Image.Image:
    """Translate a long strip section by section and stitch the results back together."""
    # the page flow translates each section; imported here because it calls back into this module
    from mangatl.pipeline.page import as_mode, resolve_pre_upscale_factor, translate_and_render

    start_time = time.time()
    width, height = strip.size
    bounds = [0, *cut_rows, height]
    log_message(
        f"Long strip ({width}x{height}): splitting into {len(bounds) - 1} sections at rows {cut_rows}",
        always_print=True,
    )

    section_config = config.updated(**{"preprocessing.split_long_strips": False})
    if config.preprocessing.auto_scale:
        # Section heights vary, so scaling each by its own area would change font sizes
        # and kernels from one section to the next. All share a nominal section's scale.
        pre_upscale = resolve_pre_upscale_factor(config.preprocessing)
        section_config = section_config.updated(
            **{
                "preprocessing.processing_scale": math.sqrt(TARGET_RATIO) * width * pre_upscale / 1000,
                "preprocessing.auto_scale": False,
            }
        )
    # Sections of one image are one page to the reader, so they always share context.
    context_count = max(config.translation.context_pages, 1)
    section_texts: list[list[str]] = []
    strip_glyph_size: list[float] = []

    def context_texts() -> list[list[str]]:
        incoming = (
            previous_context_texts_provider()
            if previous_context_texts_provider is not None
            else previous_context_texts
        )
        return [*(incoming or []), *section_texts][-context_count:]

    sections: list[Image.Image] = []
    # Open image handles can outlive a section on Windows and block directory removal.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        for index, (top, bottom) in enumerate(pairwise(bounds)):
            log_message(
                f"Processing section {index + 1}/{len(bounds) - 1} (rows {top}-{bottom})",
                always_print=True,
            )
            section_path = Path(temp_dir) / f"{image_path.stem}_section{index + 1}.png"
            strip.crop((0, top, width, bottom)).save(section_path)
            texts: list[str] = []
            section = translate_and_render(
                section_path,
                section_config,
                cancellation_manager=cancellation_manager,
                previous_context_texts_provider=context_texts,
                ocr_texts_out=texts,
                strip_glyph_size=strip_glyph_size,
                empty_bubbles_out=empty_bubbles_out,
            )
            sections.append(as_mode(section, target_mode))
            if texts:
                section_texts.append(texts)
                if ocr_texts_out is not None:
                    ocr_texts_out.extend(texts)

    stitched = Image.new(target_mode, (sections[0].width, sum(section.height for section in sections)))
    offset = 0
    for section in sections:
        stitched.paste(section, (0, offset))
        offset += section.height

    if output_path:
        save_image_with_compression(
            stitched,
            output_path,
            jpeg_quality=config.output.jpeg_quality,
            png_compression=config.output.png_compression,
            verbose=config.verbose,
        )

    log_message(
        f"Strip processing completed in {time.time() - start_time:.2f}s",
        always_print=True,
    )
    return stitched
