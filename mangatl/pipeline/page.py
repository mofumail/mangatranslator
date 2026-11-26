"""Translating one page: open, detect, clean, OCR and translate, typeset, save."""

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from PIL import Image

from mangatl.common import eval_hooks
from mangatl.common.caching import get_cache
from mangatl.common.cancellation import CancellationManager, raise_if_manager_cancelled
from mangatl.common.exceptions import ImageProcessingError
from mangatl.common.image_utils import (
    convert_image_to_target_mode,
    save_image_with_compression,
    upscale_image,
)
from mangatl.common.log import log_message
from mangatl.ml.model_manager import get_model_manager
from mangatl.pipeline import stages
from mangatl.pipeline.stages import get_image_encoding_params
from mangatl.pipeline.strip import detect_strip_text_rows, find_cut_rows, translate_strip
from mangatl.project import capture as project_capture
from mangatl.settings import PreprocessingSettings, Settings


def resolve_pre_upscale_factor(pre: PreprocessingSettings, verbose: bool = False) -> float:
    if not pre.upscale:
        return 1.0

    factor = pre.upscale_factor  # validated to 1.0-8.0
    if factor <= 1.01:
        return 1.0

    log_message(f"Initial upscaling enabled: {factor:.2f}x", verbose=verbose)
    return factor


def _apply_pre_upscale_if_needed(
    image: Image.Image,
    config: Settings,
    verbose: bool = False,
) -> tuple[Image.Image, float]:
    factor = resolve_pre_upscale_factor(config.preprocessing, verbose)
    if factor == 1.0:
        return image, 1.0

    model_type = config.preprocessing.upscale_model
    upscaled = upscale_image(image, factor, model_type=model_type, verbose=verbose)
    return upscaled, factor


def _open_source_image(image_path: Path) -> Image.Image:
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        log_message(f"Error: Input image not found at {image_path}", always_print=True)
        raise
    except Exception as e:
        log_message(f"Error opening image {image_path}: {e}", always_print=True)
        raise


def _resolve_target_mode(config: Settings, image_path: Path, output_path: str | Path | None) -> str:
    """RGB for JPEG output, RGBA for PNG, WEBP, and other formats in auto mode."""
    desired_format = config.output.format
    output_ext = Path(output_path).suffix.lower() if output_path else image_path.suffix.lower()
    if desired_format == "jpeg" or (desired_format == "auto" and output_ext in (".jpg", ".jpeg")):
        return "RGB"
    return "RGBA"


def _resolve_processing_scale(image: Image.Image, config: Settings) -> float:
    """Scale relative to a 1 MP page when auto scale is on, else the configured scale."""
    if not config.preprocessing.auto_scale:
        return config.preprocessing.processing_scale
    width, height = image.size
    processing_scale = math.sqrt((width * height) / 1_000_000)
    log_message(f"Dynamic processing scale: {processing_scale:.2f}x", verbose=config.verbose)
    return processing_scale


def _record_page(config: Settings, project_page: str | None, capture: Callable, /, **kwargs) -> None:
    """Record the page in the project file (output.project). A failure is logged, never fatal:
    the translated page is still written."""
    if config.output.project is None or project_page is None:
        return
    try:
        capture(project_capture.open_project(config.output.project), project_page, **kwargs)
    except Exception as e:
        log_message(f"Project: could not record {project_page}: {e!r}", always_print=True)


def _finish_page(
    image: Image.Image,
    config: Settings,
    output_path: str | Path | None,
    target_mode: str,
    start_time: float,
) -> Image.Image:
    """Save the page if an output path is given."""
    verbose = config.verbose
    if output_path:
        if image.mode != target_mode:
            log_message(f"Converting final image to {target_mode}", verbose=verbose)
            image = image.convert(target_mode)
        try:
            save_image_with_compression(
                image,
                output_path,
                jpeg_quality=config.output.jpeg_quality,
                png_compression=config.output.png_compression,
                verbose=verbose,
            )
        except ImageProcessingError as e:
            log_message(f"Failed to save image: {e}", always_print=True)
            raise

    log_message(f"Processing completed in {time.time() - start_time:.2f}s", always_print=True)
    return image


def translate_and_render(
    image_path: str | Path,
    config: Settings,
    output_path: str | Path | None = None,
    cancellation_manager: Optional["CancellationManager"] = None,
    previous_context_texts: list[list[str]] | None = None,
    previous_context_texts_provider: Callable[[], list[list[str]]] | None = None,
    ocr_texts_out: list[str] | None = None,
    strip_glyph_size: list[float] | None = None,
    empty_bubbles_out: list | None = None,
    project_page: str | None = None,
    project_source: tuple[Path, str, tuple[int, int, int, int]] | None = None,
):
    """
    Main function to translate manga speech bubbles and render translations using a config object.

    Args:
        image_path (str or Path): Path to input image
        config (Settings): Configuration object containing all settings.
        output_path (str or Path, optional): Path to save the final image. If None, image is not saved.
        previous_context_texts: Batch-only previous source page OCR transcripts for LLM reference
            (oldest-to-newest, one inner list per page).
        previous_context_texts_provider: Optional callback used to fetch previous-page
            OCR transcripts immediately before the translation API call.
        ocr_texts_out: Optional mutable list. When provided, the current page's OCR transcripts
            (in reading order, each joined to its translation as `source → translation`) are
            appended so the orchestrator can chain them as previous-page text context for
            subsequent pages.
        strip_glyph_size: Optional mutable list shared by the sections of one long strip. The
            first section with measurable lettering stores its glyph size, and later sections
            reuse it so auto font size stays the same along the strip.
        empty_bubbles_out: Optional mutable list. The bbox of every bubble that was cleaned
            but left without rendered text is appended, so batch runs can fail the page.
        project_page: The page's id in the project file (output.project), usually its path
            relative to the input folder; None: not recorded.
        project_source: For a strip's section: (the strip's file, the mode it was decoded in, the
            section's box), so the project stores the strip's file once.

    Returns:
        PIL.Image: Final translated image
    """
    start_time = time.time()
    eval_hooks.set_page(image_path)
    image_path = Path(image_path)
    verbose = config.verbose
    previous_context_texts = previous_context_texts or []
    empty_bubbles = empty_bubbles_out if empty_bubbles_out is not None else []
    config.request_coordinator = getattr(config, "request_coordinator", None)

    log_message(f"Using device: {config.device}", verbose=verbose)

    get_model_manager().set_hf_token(config.runtime.hf_token)

    pil_original = _open_source_image(image_path)
    image_format = pil_original.format
    mime_type, cv2_ext = get_image_encoding_params(image_format)
    log_message(f"Original image format: {image_format} -> MIME: {mime_type}", verbose=verbose)
    raise_if_manager_cancelled(cancellation_manager)

    target_mode = _resolve_target_mode(config, image_path, output_path)
    log_message(f"Target mode: {target_mode}", verbose=verbose)
    page = convert_image_to_target_mode(pil_original, target_mode, verbose)

    if config.preprocessing.split_long_strips:
        cut_rows = find_cut_rows(page, lambda crop: detect_strip_text_rows(crop, image_path, config))
        if cut_rows:
            return translate_strip(
                image_path,
                page,
                cut_rows,
                config,
                output_path,
                target_mode,
                cancellation_manager,
                previous_context_texts,
                previous_context_texts_provider,
                ocr_texts_out,
                empty_bubbles_out,
                project_page,
            )

    page, _ = _apply_pre_upscale_if_needed(page, config, verbose)
    processing_scale = _resolve_processing_scale(page, config)
    get_cache().set_current_image(page, verbose)
    work = stages.WorkPage(
        config=config,
        image=page,
        processing_scale=processing_scale,
        target_mode=target_mode,
        mime_type=mime_type,
        cv2_ext=cv2_ext,
        image_format=image_format,
        source_path=image_path,
        context_image=page.copy(),
        cancel=cancellation_manager,
    )
    translated = stages.run(
        work,
        previous_context_texts,
        previous_context_texts_provider,
        ocr_texts_out,
        strip_glyph_size,
        empty_bubbles,
    )
    _record_page(
        config,
        project_page,
        project_capture.capture,
        work=work,
        source_path=project_source[0] if project_source else image_path,
        source_crop=project_source[1:] if project_source else None,
        output_path=output_path,
    )
    return _finish_page(translated, config, output_path, target_mode, start_time)
