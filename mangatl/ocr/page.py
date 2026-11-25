"""OCR of a page's text crops: manga-ocr refined with Hayai (Japanese) or PaddleOCR-VL 1.6."""

import base64
import re
from io import BytesIO
from typing import Any

from PIL import Image

from mangatl.common.caching import get_cache
from mangatl.common.exceptions import TranslationError
from mangatl.common.log import log_message
from mangatl.ocr.readers import extract_text_with_manga_ocr, extract_text_with_paddle_ocr_vl
from mangatl.settings import Settings


def _prepare_images_for_ocr(images_b64: list[str]) -> list[Image.Image | None]:
    """Prepare base64-encoded images for OCR by decoding and converting to RGB.

    Args:
        images_b64: List of base64-encoded image strings

    Returns:
        List of PIL Images (or None for decode failures), all in RGB mode
    """
    pil_images = []
    for img_b64 in images_b64:
        try:
            image_data = base64.b64decode(img_b64)
            pil_img = Image.open(BytesIO(image_data))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)
        except Exception as e:
            log_message(
                f"Failed to decode image for manga-ocr: {e}",
                always_print=True,
            )
            pil_images.append(None)
    return pil_images


# Kana, CJK ideographs and punctuation, full-width forms (not Hangul, which uses spaces)
_CJK_CHARS = "\u3000-\u9fff\uff00-\uffef"


def _join_ocr_lines(text: str) -> str:
    """Collapse line breaks: nothing between CJK characters, a space otherwise."""
    text = re.sub(rf"(?<=[{_CJK_CHARS}])\s*\n\s*(?=[{_CJK_CHARS}])", "", text)
    return " ".join(text.split())


def _format_ocr_results(
    extracted_texts: list[str],
    bubble_metadata: list[dict[str, Any]],
) -> None:
    """Format and log OCR results.

    Args:
        extracted_texts: List of extracted text strings
        bubble_metadata: List of metadata dicts for text elements
        verbose: Whether to print verbose logging
    """
    log_lines = []

    for i, text in enumerate(extracted_texts):
        metadata = bubble_metadata[i] if i < len(bubble_metadata) else {}
        is_osb = metadata.get("is_outside_text", False)
        prefix = f"{i + 1}"
        type_label = "[OSB]" if is_osb else "[Bubble]"

        log_lines.append(f"{prefix}: {type_label} {text}")

    if log_lines:
        log_message(
            f"Raw OCR output:\n---\n{chr(10).join(log_lines)}\n---",
            always_print=True,
        )


def _check_ocr_failure(texts: list[str], provider: str | None = None) -> bool:
    """Check if all OCR results indicate failure.

    Args:
        texts: List of extracted text strings
        provider: Optional provider name for LLM OCR failure detection

    Returns:
        True if all texts indicate failure, False otherwise
    """
    if not texts:
        return True

    if provider:
        return all(f"[{provider}-OCR:" in text for text in texts)
    else:
        return all(text == "[OCR FAILED]" for text in texts)


def _perform_manga_ocr(
    images_b64: list[str],
    bubble_metadata: list[dict[str, Any]],
    debug: bool = False,
    keep_other_language: bool = False,
) -> list[str]:
    """Japanese OCR: manga-ocr, then Hayai re-reads and the other-language skip (_refine_ocr_reads)."""
    texts = _manga_ocr_reads(images_b64, bubble_metadata, debug)
    return _refine_ocr_reads(texts, images_b64, bubble_metadata, keep_other_language)


def _refine_ocr_reads(
    texts: list[str],
    images_b64: list[str],
    bubble_metadata: list[dict[str, Any]],
    keep_other_language: bool = False,
) -> list[str]:
    """Improve the manga-ocr reads of a page.

    - Outside bubbles: Hayai OCR, line by line when comic-text-detector finds several lines
      (see ocr.line_ocr).
    - Bubbles: Hayai matches manga-ocr on regular dialogue (OpenMantra ground truth: 0.9%
      character error rate for both) but reads hand-drawn moans that manga-ocr turns into
      unrelated words ("ぅ" -> "そして、"); when one read contains the other the longer is kept.
    - Pages carrying both the Japanese text and a Chinese translation of it: the Chinese block
      is marked `mt_skip` (erased, not translated) unless keep_other_language is set.
    """
    from mangatl.ml.model_manager import get_model_manager
    from mangatl.ocr import hayai as hayai_ocr
    from mangatl.ocr import line_ocr

    models_dir = get_model_manager().models_dir
    changed = False
    for i, meta in enumerate(bubble_metadata):
        if i >= len(texts) or not meta.get("is_outside_text"):
            continue
        text = line_ocr.read_region(meta, models_dir)
        if text:
            texts[i] = text
            changed = True

    bubble_idx = [i for i, m in enumerate(bubble_metadata) if i < len(texts) and not m.get("is_outside_text")]
    if bubble_idx:
        crops = _prepare_images_for_ocr([images_b64[i] for i in bubble_idx])
        ok = [(i, c) for i, c in zip(bubble_idx, crops) if c is not None]
        reads = hayai_ocr.get(models_dir).read(
            [c.convert("RGB") for _, c in ok], max_patches=512, max_new_tokens=160
        )
        for (i, _), t in zip(ok, reads):
            picked = line_ocr.pick_read(texts[i] if texts[i] != "[OCR FAILED]" else "", t.strip())
            if picked and picked != texts[i]:
                texts[i] = picked
                changed = True

    if not keep_other_language and any(line_ocr.is_japanese(t) for t in texts):
        for i, t in enumerate(texts):
            if i < len(bubble_metadata) and line_ocr.is_other_cjk(t):
                bubble_metadata[i]["mt_skip"] = True
                texts[i] = "(no text)"
                changed = True
                log_message(
                    f"Skipping block {i + 1}: not in the source language (erased, not translated)",
                    always_print=True,
                )
    if changed:
        log_message(
            "Text re-read with Hayai OCR (outside bubbles: line by line via comic-text-detector):",
            always_print=True,
        )
        _format_ocr_results(texts, bubble_metadata)
    return texts


def _manga_ocr_reads(
    images_b64: list[str],
    bubble_metadata: list[dict[str, Any]],
    debug: bool = False,
) -> list[str]:
    """Perform OCR using manga-ocr model.

    Args:
        images_b64: List of base64-encoded images
        bubble_metadata: List of metadata dicts for text elements
        debug: Whether to print verbose logging

    Returns:
        List of extracted text strings, or early return with failure list
    """
    total_elements = len(images_b64)
    log_message("Using manga-ocr for text extraction", verbose=debug)

    cache = get_cache()
    cache_key = cache.get_manga_ocr_cache_key(images_b64, total_elements)
    cached_ocr = cache.get_manga_ocr_result(cache_key)
    if cached_ocr is not None:
        if len(cached_ocr) == total_elements:
            log_message("  - Using cached manga-ocr results", always_print=True)
            return cached_ocr
        log_message("Discarding manga-ocr cache due to length mismatch", verbose=debug)

    pil_images = _prepare_images_for_ocr(images_b64)
    extracted_texts = extract_text_with_manga_ocr(pil_images, verbose=debug)

    formatted_texts = []
    for text in extracted_texts:
        if text == "[OCR FAILED]" or not text:
            formatted_texts.append(text if text else "[OCR FAILED]")
        else:
            formatted_texts.append(text)

    extracted_texts = formatted_texts

    _format_ocr_results(extracted_texts, bubble_metadata)

    if len(extracted_texts) != total_elements:
        msg = (
            f"Warning: extracted_texts length ({len(extracted_texts)}) "
            f"doesn't match total_elements ({total_elements})"
        )
        log_message(msg, always_print=True)
        while len(extracted_texts) < total_elements:
            extracted_texts.append("[OCR FAILED]")
        extracted_texts = extracted_texts[:total_elements]

    if not extracted_texts:
        log_message("manga-ocr returned empty results", verbose=debug)
        failure_results = ["[OCR FAILED]"] * total_elements
        cache.set_manga_ocr_result(cache_key, failure_results, debug)
        return failure_results

    if _check_ocr_failure(extracted_texts):
        log_message("manga-ocr returned only failures", verbose=debug)
        cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
        return extracted_texts

    cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
    return extracted_texts


def _perform_paddle_ocr_vl(
    images_b64: list[str],
    bubble_metadata: list[dict[str, Any]],
    debug: bool = False,
) -> list[str]:
    """Perform OCR using PaddleOCR-VL-1.6 model.

    Args:
        images_b64: List of base64-encoded images
        bubble_metadata: List of metadata dicts for text elements
        debug: Whether to print verbose logging

    Returns:
        List of extracted text strings, or early return with failure list
    """
    total_elements = len(images_b64)
    log_message("Using PaddleOCR-VL-1.6 for text extraction", verbose=debug)

    cache = get_cache()
    cache_key = cache.get_manga_ocr_cache_key(images_b64, total_elements, prefix="pocr16_")
    cached_ocr = cache.get_manga_ocr_result(cache_key)
    if cached_ocr is not None:
        if len(cached_ocr) == total_elements:
            log_message("  - Using cached PaddleOCR-VL-1.6 results", always_print=True)
            return cached_ocr
        log_message("Discarding PaddleOCR-VL-1.6 cache due to length mismatch", verbose=debug)

    pil_images = _prepare_images_for_ocr(images_b64)
    extracted_texts = extract_text_with_paddle_ocr_vl(pil_images, verbose=debug)

    formatted_texts = []
    for text in extracted_texts:
        if text == "[OCR FAILED]" or not text:
            formatted_texts.append(text if text else "[OCR FAILED]")
        else:
            formatted_texts.append(_join_ocr_lines(text))

    extracted_texts = formatted_texts

    _format_ocr_results(extracted_texts, bubble_metadata)

    if len(extracted_texts) != total_elements:
        msg = (
            f"Warning: extracted_texts length ({len(extracted_texts)}) "
            f"doesn't match total_elements ({total_elements})"
        )
        log_message(msg, always_print=True)
        while len(extracted_texts) < total_elements:
            extracted_texts.append("[OCR FAILED]")
        extracted_texts = extracted_texts[:total_elements]

    if not extracted_texts:
        log_message("PaddleOCR-VL-1.6 returned empty results", verbose=debug)
        failure_results = ["[OCR FAILED]"] * total_elements
        cache.set_manga_ocr_result(cache_key, failure_results, debug)
        return failure_results

    if _check_ocr_failure(extracted_texts):
        log_message("PaddleOCR-VL-1.6 returned only failures", verbose=debug)
        cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
        return extracted_texts

    cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
    return extracted_texts


def run_ocr(
    config: Settings,
    images_b64: list[str],
    bubble_metadata: list[dict[str, Any]],
    debug: bool,
) -> list[str]:
    """The OCR pass with the configured local OCR method."""
    log_message("Starting OCR step", verbose=debug)
    if config.ocr.method == "manga-ocr":
        return _perform_manga_ocr(images_b64, bubble_metadata, debug, config.ocr.keep_other_language)
    if config.ocr.method == "paddleocr-vl-1.6":
        return _perform_paddle_ocr_vl(images_b64, bubble_metadata, debug)
    raise TranslationError(f"Unknown OCR method: {config.ocr.method}")
