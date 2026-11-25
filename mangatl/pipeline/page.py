"""Translating one page: open, detect, clean, OCR and translate, typeset, save."""

import base64
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
from PIL import Image
from mangatl.common.caching import get_cache
from mangatl.common.scaling import scale_length
from mangatl.common import eval_hooks
from mangatl.common.exceptions import CancellationError, CleaningError, ImageProcessingError
from mangatl.common.log import log_message
from mangatl.clean.bubbles import clean_speech_bubbles
from mangatl.detect.bubbles import detect_panels, detect_speech_bubbles
from mangatl.common.image_utils import convert_image_to_target_mode, cv2_to_pil, pil_to_cv2, resize_to_max_side, save_image_with_compression, upscale_image, upscale_image_to_dimension
from mangatl.clean.page_glyph_map import PageGlyphMap
from mangatl.detect.reading_order import sort_bubbles_by_reading_order
from mangatl.pipeline.strip import find_cut_rows
from mangatl.ml.model_manager import get_model_manager
from mangatl.detect.outside_text import prepare_outside_text_work
from mangatl.clean.outside_text import finish_outside_text_work, paint_pending_osb_fills, process_outside_text
from mangatl.translate.service import prepare_bubble_images_for_translation
from mangatl.settings import PreprocessingSettings, Settings

from mangatl.clean.bubbles import inpaints_colored_bubbles
from mangatl.common.cancellation import CancellationManager, raise_if_manager_cancelled
from mangatl.translate.page import attach_ocr_texts, translate_page, translate_or_error_texts
from mangatl.typeset.page import (
    render_translations,
    resolve_main_max_font,
    scaled_render_sizes,
)
from mangatl.pipeline.strip import detect_strip_text_rows, translate_strip


def _should_overlap_llm_with_inpaint(config: Settings) -> bool:
    return (
        config.runtime.overlap_llm_with_inpaint
        and not config.runtime.cleaning_only
    )


def _clean_speech_bubbles_for_page(
    pil_image_processed: Image.Image,
    bubble_data: list[dict[str, Any]],
    config: Settings,
    device,
    processing_scale: float,
    verbose: bool,
    fallback_cv_image: np.ndarray,
    glyph_map: PageGlyphMap | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Clean bubbles (colored bubbles are inpainted). Returns (cv_image, info)."""
    try:
        use_otsu = config.cleaning.otsu
        if inpaints_colored_bubbles(config):
            log_message(
                "Inpainting enabled for colored bubbles",
                verbose=verbose,
            )

        cleaned_image_cv, processed_bubbles_info = clean_speech_bubbles(
            pil_image_processed,
            config.yolo_model_path,
            config.detection.confidence,
            pre_computed_detections=bubble_data,
            device=device,
            thresholding_value=config.cleaning.threshold,
            use_otsu_threshold=use_otsu,
            auto_threshold=config.cleaning.auto_threshold,
            roi_shrink_px=config.cleaning.roi_shrink_px,
            verbose=verbose,
            processing_scale=processing_scale,
            conjoined_confidence=config.detection.conjoined_confidence,
            inpaint_colored_bubbles=inpaints_colored_bubbles(config),
            seed=config.cleaning.seed,
            osb_text_verification=config.detection.outside_text.cover_with_bubbles,
            osb_text_hf_token=config.runtime.hf_token,
            inpaint_method=config.cleaning.inpainter,
            luminance_correction=config.cleaning.luminance_correction,
            bubble_detector_model=config.detection.bubble_model,
            request_coordinator=config.request_coordinator,
            glyph_map=glyph_map,
        )
        return cleaned_image_cv, processed_bubbles_info
    except CleaningError as e:
        log_message(f"Cleaning failed: {e}", always_print=True)
        return fallback_cv_image.copy(), []
    except CancellationError:
        raise
    except Exception as e:
        log_message(f"Error during cleaning: {e}", always_print=True)
        return fallback_cv_image.copy(), []


def get_image_encoding_params(pil_image_format: str | None) -> tuple[str, str]:
    """Returns (mime_type, cv2_ext) for a given PIL image format."""
    if pil_image_format and pil_image_format.upper() == "PNG":
        return "image/png", ".png"
    return "image/jpeg", ".jpg"


def _encode_context_image(
    image: Image.Image, config: Settings, cv2_ext: str
) -> str:
    """Resize a page for LLM context per the context upscale setting, base64-encoded."""
    verbose = config.verbose
    # processing_scale is intentionally not used for context_image_max_side_pixels
    max_side = scale_length(
        config.translation.context_image_max_side,
        None,
        minimum=512,
        maximum=4096,
    )
    upscale_method = config.ocr.upscale

    if upscale_method in ("model", "model_lite"):
        model_manager = get_model_manager()
        if upscale_method == "model":
            upscale_model = model_manager.load_upscale(verbose=verbose)
        else:
            upscale_model = model_manager.load_upscale_lite(verbose=verbose)
        image = upscale_image_to_dimension(
            upscale_model,
            image,
            max_side,
            config.device,
            "max",
            upscale_method,
            verbose,
        )
        image = resize_to_max_side(image, max_side, verbose=verbose)
        model_manager.clear_cache()
    elif upscale_method == "lanczos":
        image = resize_to_max_side(image, max_side, verbose=verbose, antiring=True)

    is_success, buffer = cv2.imencode(cv2_ext, pil_to_cv2(image))
    if not is_success:
        raise ImageProcessingError(f"Context image encoding to {cv2_ext} failed")
    return base64.b64encode(buffer).decode("utf-8")


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


def as_mode(image: Image.Image, mode: str) -> Image.Image:
    return image if image.mode == mode else image.convert(mode)


def _open_source_image(image_path: Path) -> Image.Image:
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        log_message(f"Error: Input image not found at {image_path}", always_print=True)
        raise
    except Exception as e:
        log_message(f"Error opening image {image_path}: {e}", always_print=True)
        raise


def _resolve_target_mode(
    config: Settings, image_path: Path, output_path: str | Path | None
) -> str:
    """RGB for JPEG output, RGBA for PNG, WEBP, and other formats in auto mode."""
    desired_format = config.output.format
    output_ext = (
        Path(output_path).suffix.lower() if output_path else image_path.suffix.lower()
    )
    if desired_format == "jpeg" or (
        desired_format == "auto" and output_ext in (".jpg", ".jpeg")
    ):
        return "RGB"
    return "RGBA"


def _resolve_processing_scale(
    image: Image.Image, config: Settings
) -> float:
    """Scale relative to a 1 MP page when auto scale is on, else the configured scale."""
    if not config.preprocessing.auto_scale:
        return config.preprocessing.processing_scale
    width, height = image.size
    processing_scale = math.sqrt((width * height) / 1_000_000)
    log_message(
        f"Dynamic processing scale: {processing_scale:.2f}x", verbose=config.verbose
    )
    return processing_scale


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

    log_message(
        f"Processing completed in {time.time() - start_time:.2f}s", always_print=True
    )
    return image


def _detect_page_bubbles(
    image_path: Path, image: Image.Image, config: Settings
) -> tuple[list[dict[str, Any]], list]:
    """Returns (bubbles, text-free boxes), both empty if detection fails."""
    log_message("Detecting speech bubbles...", verbose=config.verbose)
    try:
        return detect_speech_bubbles(
            image_path,
            config.yolo_model_path,
            config.detection.confidence,
            verbose=config.verbose,
            device=config.device,
            conjoined_detection=config.detection.conjoined,
            conjoined_confidence=config.detection.conjoined_confidence,
            image_override=image,
            osb_enabled=config.detection.outside_text.enabled,
            osb_text_verification=config.detection.outside_text.cover_with_bubbles,
            osb_text_hf_token=config.runtime.hf_token,
            bubble_detector_model=config.detection.bubble_model,
        )
    except Exception as e:
        log_message(f"Error during detection: {e}", always_print=True)
        return [], []


def _detect_page_panels(
    image_path: Path, image: Image.Image, config: Settings
) -> list | None:
    """Panels for reading order, or None when panel sorting is off or detection fails."""
    if not config.detection.panel_sorting:
        return None

    verbose = config.verbose
    try:
        log_message("Detecting panels...", verbose=verbose)
        panels = detect_panels(
            image_path,
            confidence=config.detection.panel_confidence,
            device=config.device,
            verbose=verbose,
            image_override=image,
        )
        if panels:
            log_message(f"Detected {len(panels)} panels", always_print=True)
        else:
            log_message("No panels detected", verbose=verbose)
    except Exception as e:
        log_message(
            f"Panel detection failed: {e}. Using global sorting.", always_print=True
        )
        panels = None
    return panels


def _encode_full_page_context(
    image: Image.Image, config: Settings, cv2_ext: str
) -> str | None:
    try:
        encoded = _encode_context_image(image, config, cv2_ext)
    except Exception as e:
        log_message(
            f"Warning: Failed to encode full image context: {e}", always_print=True
        )
        return None
    log_message("Encoded full image for context", verbose=config.verbose)
    return encoded


def _clean_page(
    image: Image.Image,
    bubble_data: list[dict[str, Any]],
    config: Settings,
    processing_scale: float,
    target_mode: str,
    glyph_map: PageGlyphMap | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Returns (page with bubbles cleaned, in target_mode, per-bubble cleaning info).

    `glyph_map` is the text map of the page before outside text was inpainted, which
    still holds inside the bubbles.
    """
    if not bubble_data:
        return as_mode(image, target_mode), []
    log_message("Cleaning speech bubbles...", verbose=config.verbose)
    cleaned_cv, processed_bubbles_info = _clean_speech_bubbles_for_page(
        image,
        bubble_data,
        config,
        config.device,
        processing_scale,
        config.verbose,
        pil_to_cv2(image),
        glyph_map,
    )
    return as_mode(cv2_to_pil(cleaned_cv), target_mode), processed_bubbles_info


def _finish_deferred_cleaning(
    outside_work,
    image: Image.Image,
    bubble_data: list[dict[str, Any]],
    config: Settings,
    processing_scale: float,
    target_mode: str,
    glyph_map: PageGlyphMap | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Finish the deferred outside text inpainting, then clean the bubbles on top of it."""
    if outside_work is not None:
        image, _ = finish_outside_text_work(outside_work)
    return _clean_page(
        image, bubble_data, config, processing_scale, target_mode, glyph_map
    )


def _prepare_bubble_crops(
    bubble_data: list[dict[str, Any]],
    processed_bubbles_info: list[dict[str, Any]],
    page_cv: np.ndarray,
    config: Settings,
    mime_type: str,
) -> list[dict[str, Any]]:
    """Attach the refined cleaning masks to the bubbles and encode their crops for the LLM."""
    verbose = config.verbose
    log_message("Preparing bubble images...", verbose=verbose)

    mask_by_bbox: dict[tuple, Any] = {}
    for info in processed_bubbles_info:
        bbox_key = tuple(round(v) for v in info.get("bbox", ()))
        if len(bbox_key) != 4:
            continue
        mask = info.get("mask")
        if mask is None:
            mask = info.get("base_mask")
        if mask is not None:
            mask_by_bbox[bbox_key] = mask
    for bubble in bubble_data:
        bbox_key = tuple(round(v) for v in bubble.get("bbox", ()))
        if bbox_key in mask_by_bbox:
            bubble["seg_mask"] = mask_by_bbox[bbox_key]

    upscale_method = config.ocr.upscale

    model_manager = get_model_manager()
    upscale_model = None
    if upscale_method == "model":
        upscale_model = model_manager.load_upscale(verbose=verbose)
    elif upscale_method == "model_lite":
        upscale_model = model_manager.load_upscale_lite(verbose=verbose)

    bubble_data = prepare_bubble_images_for_translation(
        bubble_data,
        page_cv,
        upscale_model,
        config.device,
        mime_type,
        config.ocr.bubble_min_side,
        upscale_method,
        config.ocr.whiteout_neighbors,
        verbose,
    )
    if upscale_model is not None:
        model_manager.clear_cache()

    action = "Upscaled" if upscale_method != "none" else "Prepared"
    log_message(
        f"{action} {len(bubble_data)} bubble images for translation",
        always_print=True,
    )
    return bubble_data


def _sort_text_elements(
    valid_bubble_data: list[dict[str, Any]],
    outside_text_data: list[dict[str, Any]],
    config: Settings,
    panels: list | None,
) -> list[dict[str, Any]]:
    """Sort speech bubbles and outside text together into reading order."""
    reading_direction = config.translation.reading_direction
    if outside_text_data:
        log_message(
            f"Including {len(outside_text_data)} outside text regions in reading order calculation",
            verbose=config.verbose,
        )
    log_message(
        f"Sorting all text elements ({reading_direction.upper()})",
        verbose=config.verbose,
    )
    return sort_bubbles_by_reading_order(
        valid_bubble_data + outside_text_data, reading_direction, panels=panels
    )


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

    Returns:
        PIL.Image: Final translated image
    """
    start_time = time.time()
    eval_hooks.set_page(image_path)
    image_path = Path(image_path)
    verbose = config.verbose
    previous_context_texts = previous_context_texts or []
    empty_bubbles = empty_bubbles_out if empty_bubbles_out is not None else []
    config.request_coordinator = getattr(
        config, "request_coordinator", None
    )

    log_message(f"Using device: {config.device}", verbose=verbose)

    get_model_manager().set_hf_token(config.runtime.hf_token)

    pil_original = _open_source_image(image_path)
    image_format = pil_original.format
    mime_type, cv2_ext = get_image_encoding_params(image_format)
    log_message(
        f"Original image format: {image_format} -> MIME: {mime_type}", verbose=verbose
    )
    raise_if_manager_cancelled(cancellation_manager)

    target_mode = _resolve_target_mode(config, image_path, output_path)
    log_message(f"Target mode: {target_mode}", verbose=verbose)
    page = convert_image_to_target_mode(pil_original, target_mode, verbose)

    if config.preprocessing.split_long_strips:
        cut_rows = find_cut_rows(
            page, lambda crop: detect_strip_text_rows(crop, image_path, config)
        )
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
            )

    page, _ = _apply_pre_upscale_if_needed(page, config, verbose)

    processing_scale = _resolve_processing_scale(page, config)
    get_cache().set_current_image(page, verbose)
    full_page_context_source = page.copy()

    # Detect speech bubbles first so OSB processing can respect bubble regions
    bubble_data, text_free_boxes = _detect_page_bubbles(image_path, page, config)
    raise_if_manager_cancelled(cancellation_manager)
    panels = _detect_page_panels(image_path, page, config)
    raise_if_manager_cancelled(cancellation_manager)
    # Built on first use from the original page, then shared by OSB and bubble cleaning
    glyph_map = PageGlyphMap.for_page(page, config, verbose)

    # Outside text is always detected here, but with overlap its inpainting (and bubble
    # cleaning) is deferred to run alongside the LLM call
    overlap = _should_overlap_llm_with_inpaint(config)
    outside_work = None
    if overlap:
        outside_work = prepare_outside_text_work(
            page,
            config,
            image_path,
            image_format,
            verbose,
            bubble_data=bubble_data,
            text_free_boxes=text_free_boxes,
            panels=panels,
            glyph_map=glyph_map,
        )
        outside_text_data = (
            outside_work.outside_text_data if outside_work is not None else []
        )
    else:
        page, outside_text_data = process_outside_text(
            page,
            config,
            image_path,
            image_format,
            verbose,
            bubble_data=bubble_data,
            text_free_boxes=text_free_boxes,
            panels=panels,
            glyph_map=glyph_map,
        )
    # With overlap this is the pre-inpaint page, which the bubble and OSB LLM crops use
    page_cv = pil_to_cv2(page)

    full_image_b64 = None
    if config.translation.full_page_context:
        full_image_b64 = _encode_full_page_context(
            full_page_context_source, config, cv2_ext
        )
    raise_if_manager_cancelled(cancellation_manager)

    if not bubble_data and not outside_text_data:
        log_message("No speech bubbles or outside text detected", always_print=True)
        return _finish_page(page, config, output_path, target_mode, start_time)
    if bubble_data:
        log_message(f"Detected {len(bubble_data)} bubbles", verbose=verbose)
    if outside_text_data:
        log_message(
            f"Detected {len(outside_text_data)} outside text regions", verbose=verbose
        )

    if overlap:
        cleaned, processed_bubbles_info = as_mode(page, target_mode), []
    else:
        cleaned, processed_bubbles_info = _clean_page(
            page, bubble_data, config, processing_scale, target_mode, glyph_map
        )

    if config.runtime.cleaning_only:
        log_message("Cleaning only mode - skipping translation", always_print=True)
        cleaned = paint_pending_osb_fills(cleaned, outside_text_data)
        return _finish_page(cleaned, config, output_path, target_mode, start_time)

    sizes = scaled_render_sizes(config, processing_scale)
    page_glyphs: dict = {}  # this page's lettering sizes, for bigger text in shouted bubbles
    bubble_data = _prepare_bubble_crops(
        bubble_data, processed_bubbles_info, page_cv, config, mime_type
    )
    finish_deferred_cleaning = partial(
        _finish_deferred_cleaning,
        outside_work,
        page,
        bubble_data,
        config,
        processing_scale,
        target_mode,
        glyph_map,
    )
    valid_bubble_data = [b for b in bubble_data if b.get("image_b64")]
    if not valid_bubble_data and not outside_text_data:
        log_message(
            "No valid bubble images or outside text for translation", always_print=True
        )
        if overlap:
            cleaned, _ = finish_deferred_cleaning()
        return _finish_page(cleaned, config, output_path, target_mode, start_time)
    raise_if_manager_cancelled(cancellation_manager)

    sorted_bubble_data = _sort_text_elements(
        valid_bubble_data, outside_text_data, config, panels
    )

    bubble_images_b64 = [
        bubble["image_b64"] for bubble in sorted_bubble_data if "image_b64" in bubble
    ]
    bubble_mime_types = [
        bubble["mime_type"]
        for bubble in sorted_bubble_data
        if "image_b64" in bubble and "mime_type" in bubble
    ]
    translated_texts: list[str] = []
    current_ocr_texts: list[str] = []
    if not bubble_images_b64:
        log_message("No valid bubbles after sorting", always_print=True)
        if overlap:
            cleaned, processed_bubbles_info = finish_deferred_cleaning()
    else:
        call_api = partial(
            translate_page,
            config,
            sorted_bubble_data,
            bubble_images_b64,
            bubble_mime_types,
            full_image_b64,
            mime_type,
            previous_context_texts,
            previous_context_texts_provider,
        )
        translate = partial(
            translate_or_error_texts,
            sorted_bubble_data=sorted_bubble_data,
            image_count=len(bubble_images_b64),
            config=config,
            ocr_texts_out=ocr_texts_out,
        )
        if overlap:
            log_message(
                "Running LLM translation concurrently with inpainting",
                always_print=True,
            )
            with ThreadPoolExecutor(max_workers=2) as overlap_executor:
                inpaint_future = overlap_executor.submit(finish_deferred_cleaning)
                translate_future = overlap_executor.submit(call_api)
                try:
                    cleaned, processed_bubbles_info = inpaint_future.result()
                except Exception:
                    translate_future.cancel()
                    raise
                translated_texts, current_ocr_texts = translate(translate_future.result)
        else:
            translated_texts, current_ocr_texts = translate(call_api)

    if current_ocr_texts:
        attach_ocr_texts(sorted_bubble_data, current_ocr_texts, verbose)

    sizes = sizes._replace(
        main_max_font=resolve_main_max_font(
            config,
            page_cv,
            processed_bubbles_info,
            sizes.main_min_font,
            sizes.main_max_font,
            processing_scale,
            strip_glyph_size,
            verbose,
            page_glyphs,
        )
    )
    cleaned = render_translations(
        cleaned,
        sorted_bubble_data,
        translated_texts,
        processed_bubbles_info,
        page_cv,
        config,
        sizes,
        processing_scale,
        empty_bubbles,
        cancellation_manager,
        page_glyphs,
    )
    cleaned = paint_pending_osb_fills(cleaned, outside_text_data)
    return _finish_page(cleaned, config, output_path, target_mode, start_time)
