"""Translating a page: the translation request, then the fix-ups and the second look at short items."""

import re
from collections.abc import Callable
from typing import Any
from mangatl.common.caching import get_cache
from mangatl.typeset.narration import mark_narration
from mangatl.translate.vision_pass import revise_short_items
from mangatl.common import eval_hooks
from mangatl.common.exceptions import CancellationError, TranslationError
from mangatl.common.log import log_message
from mangatl.translate.service import pair_context_lines, call_translation_api_batch
from mangatl.settings import PROVIDER, Settings
from mangatl.typeset.page import INVALID_TRANSLATION_VALUES


# Translation errors that fail the page instead of leaving error text in its bubbles.
# utils/endpoints/_http.py words its errors to match these.
CRITICAL_TRANSLATION_ERRORS = (
    "429",
    "rate limit",
    "rate-limit",
    "auth",
    "unauthorized",
    "forbidden",
    "payment",
    "quota",
    "empty response",
    "api failed",
)


def translate_page(
    config: Settings,
    sorted_bubble_data: list[dict[str, Any]],
    images_b64: list[str],
    mime_types: list[str],
    full_image_b64: str | None,
    full_image_mime_type: str,
    previous_context_texts: list[list[str]],
    previous_context_texts_provider: Callable[[], list[list[str]]] | None,
) -> tuple[list[str], list[str]]:
    """Returns (translations, OCR transcripts) for the sorted page elements."""
    ocr_texts: list[str] = []
    if previous_context_texts_provider is not None:
        previous_context_texts = previous_context_texts_provider() or []

    context_parts = []
    usable_context_text_pages = sum(
        1
        for page_texts in previous_context_texts
        if any(
            (text or "").strip() and (text or "").strip() != "[OCR FAILED]"
            for text in (page_texts or [])
        )
    )
    if usable_context_text_pages:
        context_parts.append(
            f"Previous Context OCR Text: {usable_context_text_pages} page(s)"
        )
    context_suffix = f" ({', '.join(context_parts)})" if context_parts else ""
    log_message(
        f"Translating {len(images_b64)} bubbles: "
        f"{config.translation.source_language} → {config.translation.target_language}"
        f"{context_suffix}",
        always_print=True,
    )
    page_b64 = full_image_b64 or ""
    page_mime = full_image_mime_type if full_image_b64 else "image/jpeg"

    def translate(ocr_out: list[str]) -> list[str]:
        return call_translation_api_batch(
            config=config,
            images_b64=images_b64,
            full_image_b64=page_b64,
            mime_types=mime_types,
            full_image_mime_type=page_mime,
            bubble_metadata=sorted_bubble_data,
            previous_context_texts=previous_context_texts,
            ocr_texts_output=ocr_out,
            debug=config.verbose,
        )

    texts = translate(ocr_texts)
    texts = _retry_truncated(texts, sorted_bubble_data, ocr_texts, translate)
    _mark_failed_and_skipped(texts, sorted_bubble_data)
    mark_narration(texts, sorted_bubble_data)
    texts = revise_short_items(config, sorted_bubble_data, ocr_texts, texts,
                               page_b64, page_mime)
    eval_hooks.dump_text(sorted_bubble_data, ocr_texts, texts)
    return texts, ocr_texts


def _content_len(text: str) -> int:
    return sum(1 for c in text if c.isalnum())


def _retry_truncated(
    texts: list[str],
    sorted_bubble_data: list[dict[str, Any]],
    ocr_texts: list[str],
    translate: Callable[[list[str]], list[str]],
) -> list[str]:
    """The LLM occasionally returns only the first sentence of a long block (seen: a 120-char
    narration block came back as "-A little while later."), silently dropping the dialogue.
    If a long source line gets an implausibly short translation, translate the page once more
    (bypassing the translation cache) and keep, per line, the fuller of the two answers."""

    def looks_truncated(src: str, tr: str) -> bool:
        n = _content_len(src)
        return n >= 30 and isinstance(tr, str) and _content_len(tr) < 0.5 * n

    sources = [m.get("ocr_text") or (ocr_texts[i] if i < len(ocr_texts) else "")
               for i, m in enumerate(sorted_bubble_data)]
    bad = [i for i, (s, t) in enumerate(zip(sources, texts)) if looks_truncated(s, t)]
    if not bad:
        return texts
    log_message(f"Translation looks truncated for item(s) {[i + 1 for i in bad]}; retrying once",
                always_print=True)
    get_cache().clear_translation_cache()
    again = translate([])
    for i in bad:
        if i < len(again) and _content_len(again[i]) > _content_len(texts[i]):
            texts[i] = again[i]
    return texts


def _mark_failed_and_skipped(texts: list[str], sorted_bubble_data: list[dict[str, Any]]) -> None:
    for i, t in enumerate(texts):
        # Items the LLM skipped come back as "[<provider>-Translate: missing item N]", which the
        # renderer doesn't recognize as an error and would draw onto the page. Mark them as
        # failed instead, so the original lettering is restored.
        if isinstance(t, str) and re.fullmatch(r"\s*\[[^\]]*missing item \d+\]\s*", t, re.I):
            texts[i] = "[OCR FAILED]"
    for i, m in enumerate(sorted_bubble_data):
        if m.get("mt_skip") and i < len(texts):
            texts[i] = ""  # other-language block: rendered as "erase only"


def translate_or_error_texts(
    get_result: Callable[[], tuple[list[str], list[str]]],
    sorted_bubble_data: list[dict[str, Any]],
    image_count: int,
    config: Settings,
    ocr_texts_out: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Run the translation; non-critical failures become per-bubble error text."""
    try:
        texts, ocr_texts = get_result()
    except TranslationError as e:
        error_str = str(e).lower()
        if any(t in error_str for t in CRITICAL_TRANSLATION_ERRORS):
            raise
        log_message(f"Translation failed: {e}", always_print=True)
        texts = [f"[Translation Error: {e}]"] * image_count
        ocr_texts = []
    except CancellationError:
        raise
    except Exception as e:
        log_message(f"Translation API error: {e}", always_print=True)
        texts = [
            "[Translation Error: API call raised exception]" for _ in sorted_bubble_data
        ]
        ocr_texts = []

    if ocr_texts and ocr_texts_out is not None:
        ocr_texts_out.extend(pair_context_lines(ocr_texts, texts))

    provider_tag = f"[{PROVIDER}:"
    valid_translations = [
        t
        for t in texts
        if t
        and not t.startswith("[Translation Error")
        and not t.startswith("API Error")
        and not t.startswith(provider_tag)
        and t.strip() not in INVALID_TRANSLATION_VALUES
    ]
    if image_count and not valid_translations:
        raise TranslationError("All bubbles failed.")
    return texts, ocr_texts


def attach_ocr_texts(
    sorted_bubble_data: list[dict[str, Any]], ocr_texts: list[str], verbose: bool
) -> None:
    if len(ocr_texts) == len(sorted_bubble_data):
        for bubble, ocr_text in zip(sorted_bubble_data, ocr_texts, strict=True):
            bubble["ocr_text"] = ocr_text
    else:
        log_message(
            "OCR/translation count mismatch; OSB unchanged-text restore disabled",
            verbose=verbose,
        )
