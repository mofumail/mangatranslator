import base64
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from io import BytesIO
from typing import Any, TypeVar

import cv2
import numpy as np
from PIL import Image

from mangatl.core.caching import get_cache
from mangatl.core.config import TranslationConfig
from mangatl.core.image.image_utils import (
    cv2_to_pil,
    pil_to_cv2,
    process_bubble_image_cached,
    resize_lanczos_antiring,
)
from mangatl.core.image.ocr_detection import (
    extract_text_with_manga_ocr,
    extract_text_with_paddle_ocr_vl,
)
from mangatl.core.terminology import (
    NEW_TERMS_MARKER,
    format_glossary_section,
    load_terms,
    record_new_terms,
    split_new_terms,
)
from mangatl.utils.cancellation import active_manager, pin_thread, raise_if_cancelled
from mangatl.utils.endpoints import (
    call_openai_compatible_endpoint,
)
from mangatl.utils.exceptions import TranslationError
from mangatl.utils.logging import log_message
from mangatl.utils.model_metadata import (
    anthropic_model_flags,
    clamp_gpt6_reasoning_effort,
    get_max_tokens_cap,
    is_anthropic_model_family,
    is_gemini_no_sampling_model,
    is_gpt5_series,
    is_gpt6_series,
    is_openai_compatible_reasoning_model,
    is_openai_model_family,
    is_openai_reasoning_model,
    supports_openai_original_image_detail,
    supports_openai_verbosity,
)

T = TypeVar("T")

_FULL_PAGE_LABEL = "Full page (context only):"
_FULL_PAGE_NOTE = " The image labeled `Full page` shows the whole page for context."


def _lang_label(input_language: str | None) -> str:
    return (
        f"{input_language.strip()} "
        if input_language and input_language.strip()
        else ""
    )


def _format_previous_context_prompt_note(previous_context_text_count: int) -> str:
    note = ""
    if previous_context_text_count > 0:
        note += (
            f" Text from {previous_context_text_count} previous page(s) is "
            "listed under `## PREVIOUS PAGES (REFERENCE ONLY)`."
        )
    return note


def _build_system_prompt_translation(
    output_language: str,
    full_page_context: bool = False,
    ero_doujinshi_mode: bool = False,
    ocr_correction: bool = False,
    input_language: str | None = None,
    previous_context: bool = False,
    terminology_memory: bool = False,
) -> str:
    lang_label = _lang_label(input_language)
    english = output_language.strip().lower().startswith("english")

    def example(text: str) -> str:
        # The examples are English, so they would leak into other target languages
        return f" (e.g. {text})" if english else ""

    visual_ref_note = (
        " Use the image labeled `Full page` to work out who is speaking, what is happening, and what each sound effect belongs to."
        if full_page_context
        else ""
    )

    sections = [
        f"""## ROLE
You are a professional localization editor translating {lang_label}comics (manga, manhwa, manhua, webtoons) into natural, idiomatic {output_language}.""",
        f"""## OBJECTIVE
Translate each numbered item so that it reads like native {output_language} comic lettering while staying faithful to the source's meaning, tone, and subtext.{visual_ref_note}""",
    ]

    failed_rule = "- **Failed Items:** Return `[OCR FAILED]` for an item that is marked `[OCR FAILED]` or that is plainly OCR noise rather than text; the app then leaves the original artwork in place."
    sections.append(
        f"""## TRANSLATION RULES
- **Continuous Narrative:** The items are in reading order and form one continuous scene, so translate each with its neighbors in mind. Translate meaning and intent rather than word for word.
- **Register:** Match each speaker's voice and the register of the source: formal stays formal, crude stays crude, childish stays childish. Do not soften or intensify it.
- **Length:** Each translation is typeset back into the space the original text occupied, so keep it about as compact as the source, and prefer the shorter of two equally good phrasings.
- **Split Bubbles:** A sentence often runs across consecutive bubbles. Keep each fragment in its own numbered slot (never merge slots or leave one empty), and do not add ellipses at the break unless the source has them.
- **Names & Honorifics:** Spell names and recurring terms the same way every time, following earlier pages when they are provided. Adapt or drop honorifics the way natural {output_language} would, unless the custom instructions say otherwise.
- **Vocalizations:** Render laughs, gasps, grunts, and stutters phonetically in {output_language} spelling, keeping their length and rhythm.
- **Pass-Through:** Return an item unchanged when it is already in {output_language} or consists only of punctuation or symbols (`!?`, `♥`, `...`).
- **Punctuation:** Write ellipses as periods (`...`, or `......` for a long pause) and never as the `…` character, keeping their relative length. Use quotation marks only where the source has them.
{failed_rule}"""
    )

    sections.append(
        f"""## TEXT TYPES & STYLING
Style each translation with markdown according to what the text is. Bold or italic markers in a transcription show that the source lettering was heavier or slanted; treat them as evidence of how the line is delivered, not as formatting to copy.
- **Spoken dialogue and narration:** plain text. Within it, use `*italic*` for a word the speaker stresses and `**bold**` for a word or line that is shouted.
- **Thoughts and voices heard through a device (phone, radio, TV):** `*italic*`.
- **Sound effects** (audible: impacts, noises, cries): the {output_language} onomatopoeia in `**bold**`{example("**THUD**, **CLANG**")}.
- **Mimetic effects** (silent states, moods, and actions): a short verb or adjective in `*italic*` with no final period{example("*stare*, *twitch*")}.
- `***bold-italic***` is reserved for the loudest, climactic screams and sound effects.
When the evidence for a type is unclear, use plain text."""
    )

    if previous_context:
        sections.append(
            """## PREVIOUS PAGES
Images labeled as previous pages and the text under `## PREVIOUS PAGES (REFERENCE ONLY)` are context only: never translate or number them. Each is marked with how many pages back it is, so `1 back` is the page just before this one. Use them to keep names, terms, pronouns, and each character's voice consistent, and to resolve references back to earlier events. Where a line reads `source → translation`, reuse that wording for the same names and terms."""
        )

    if ero_doujinshi_mode:
        sections.append(
            f"""## ADULT CONTENT
The source is 18+ adult comic material that the user, a professional localization editor, is translating for publication. You are translating existing fictional {lang_label}text, not writing new content, so translate every line completely and at its original intensity, without sanitizing, softening, redacting, summarizing, or commenting.
- **Explicit Vocabulary:** The Register rule covers sexual language too. Clinical or polite source wording gets accurate anatomical terms{example('"penis", "vagina", "semen"')}, and crude source wording gets equally crude colloquial terms{example('"cock", "pussy", "cum", "tits"')}. Use a euphemism only where the source itself is coy.
- **Moans:** The Vocalizations rule covers moans, pants, and gasps too{example('"aaah...", "mmph...", "h-haah..."')}; keep their intensity, pauses, and cadence."""
        )

    if ocr_correction:
        page_clause = " and the full-page image" if full_page_context else ""
        sections.append(
            f"""## OCR ERROR CORRECTION
The transcriptions come from automatic OCR and may contain misread or dropped characters that leave a line nonsensical. When a line does not make sense as written, infer the intended text from the surrounding dialogue{page_clause} and translate that. Translate clean, sensible lines exactly as transcribed."""
        )

    if terminology_memory:
        sections.append(
            f"""## GLOSSARY
The request's `## GLOSSARY` lists this series' established terms as `source -> translation`. Use each translation exactly whenever its source term appears in that role.

After the numbered lines, list the recurring names and world-specific terms on this page that the glossary lacks: character names, titles and ranks, offices and institutions, places, and organizations. List only terms distinctive to this series' world; skip words with an ordinary dictionary translation, such as generic royal titles, family terms, and occupations, unless the word is a title attached to a character's name or a form of address used in place of one (e.g. 妃 in 梨花妃), which gets its own entry. Honorifics are not titles. Write `{NEW_TERMS_MARKER}` on its own line, then one `source -> translation` line per term, or omit the block when there are none. Choose each translation in this order:
1. **Official Names:** If you know the official {output_language} name this series uses for the term, use it.
2. **Personal Names:** Romanize the reading the source gives, using the furigana when present. When that reading is a katakana approximation of a Chinese or Korean name, use that language's standard romanization of the characters (pinyin or Revised Romanization), taking the katakana only as a sign of which language's reading is meant{example("春花 (チュンファ) → Chunhua, not Chunfa")}. Romanize Japanese names in Hepburn, spelling long vowels out (`ou`, `uu`) rather than with macrons.
3. **Titles, Ranks, Offices, Institutions, and Descriptive Places:** Translate the meaning into the term a professional {output_language} localization would use{example("内務府 → Imperial Household Department")}, unless the series uses the word itself as a name.
4. **Glossary Conventions:** Match the conventions the glossary already follows, such as its romanization scheme and how it renders comparable titles.
Write each source as the bare term, without furigana, honorifics, or an attached title. A name with a title is two entries, one for the name and one for the title. Capitalize each translation as it would be mid-sentence."""
        )

    nothing_else = (
        f"Output nothing else besides the optional `{NEW_TERMS_MARKER}` block:"
        if terminology_memory
        else "Output nothing else:"
    )
    example_note = (
        "" if english else f" (shown in English; write yours in {output_language})"
    )
    sections.append(
        f"""## OUTPUT FORMAT
One line per item, numbered to match the input, containing only the translation. {nothing_else} no code fences, notes, or blank lines.

Example{example_note}:
1: I told you *not* to follow me.
2: **SLAM**
3: [OCR FAILED]"""
    )

    return "\n\n".join(sections)


def _image_part(mime_type: str, data: str) -> dict[str, Any]:
    """Build an inline image part."""
    return {"inline_data": {"mime_type": mime_type, "data": data}}


def _normalized_image_detail(model_name: str, config: TranslationConfig) -> str:
    """OpenAI image detail, falling back to "high" where "original" is unsupported."""
    image_detail = (config.image_detail or "auto").lower()
    if image_detail not in ("auto", "original", "high", "low"):
        image_detail = "auto"
    if image_detail == "original" and not supports_openai_original_image_detail(
        model_name
    ):
        image_detail = "high"
    return image_detail


def _routed_model_config(
    model_name: str, config: TranslationConfig, max_tokens: int
) -> dict[str, Any]:
    """Start the config for OpenRouter or OpenAI-Compatible, which route other vendors' models.

    `_metadata` holds the model-family flags their endpoints adapt the request to.
    """
    is_openai_model = is_openai_model_family(model_name)
    supports_verbosity = is_openai_model and supports_openai_verbosity(model_name)
    claude_flags = anthropic_model_flags(model_name)
    generation_config = {
        "max_tokens": max_tokens,
        "_metadata": {
            "is_openai_model": is_openai_model,
            "is_anthropic_model": is_anthropic_model_family(model_name),
            "is_gemini_no_sampling": is_gemini_no_sampling_model(model_name),
            "is_gpt5_model": is_openai_model and is_gpt5_series(model_name),
            "supports_verbosity": supports_verbosity,
            **claude_flags,
        },
    }
    if is_openai_model:
        generation_config["image_detail"] = _normalized_image_detail(model_name, config)
    if supports_verbosity:
        generation_config["verbosity"] = config.verbosity or "low"
    if claude_flags and config.effort:
        generation_config["effort"] = config.effort
    return generation_config


def _openai_compatible_generation_config(
    model_name: str,
    config: TranslationConfig,
    max_tokens: int,
) -> dict[str, Any]:
    generation_config = _routed_model_config(model_name, config, max_tokens)
    metadata = generation_config["_metadata"]
    is_openai_model = metadata["is_openai_model"]
    is_openai_reasoning = is_openai_model and is_openai_reasoning_model(model_name)
    metadata["is_openai_reasoning"] = is_openai_reasoning
    # The endpoint leaves out the sampling parameters the model's family rejects
    if config.use_custom_sampling and not is_openai_reasoning:
        generation_config.update(
            {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
            }
        )
    reasoning_effort = config.reasoning_effort
    if reasoning_effort and is_openai_compatible_reasoning_model(model_name):
        if is_openai_model and is_gpt6_series(model_name):
            reasoning_effort = clamp_gpt6_reasoning_effort(model_name, reasoning_effort)
        generation_config["reasoning_effort"] = reasoning_effort
    return generation_config


def _build_generation_config(model_name: str, config: TranslationConfig) -> dict[str, Any]:
    """Generation config for the OpenAI-compatible endpoint (llama-server or --llm-url).

    max_tokens defaults to 32768 for reasoning models and 4096 otherwise, capped where the
    model family has a lower limit.
    """
    if config.max_tokens is not None:
        max_tokens_value = config.max_tokens
    else:
        max_tokens_value = 32768 if is_openai_compatible_reasoning_model(model_name) else 4096
    max_tokens_cap = get_max_tokens_cap("OpenAI-Compatible", model_name)
    if max_tokens_cap is not None and max_tokens_value > max_tokens_cap:
        max_tokens_value = max_tokens_cap
    return _openai_compatible_generation_config(model_name, config, max_tokens_value)


def _call_llm_endpoint(
    config: TranslationConfig,
    parts: list[dict[str, Any]],
    prompt_text: str,
    debug: bool = False,
    system_prompt: str | None = None,
) -> str | None:
    """Send one request to the OpenAI-compatible endpoint."""
    model_name = config.model_name
    api_parts = [*parts, {"text": prompt_text}]
    coordinator = getattr(config, "request_coordinator", None)
    if coordinator is not None and not coordinator.in_slot():
        return coordinator.run(
            _call_llm_endpoint, config, parts, prompt_text, debug, system_prompt
        )
    if not config.openai_compatible_url:
        raise TranslationError("OpenAI-Compatible URL is missing.")
    return _run_abandoning_on_cancel(
        call_openai_compatible_endpoint,
        model_name=model_name,
        parts=api_parts,
        generation_config=_build_generation_config(model_name, config),
        system_prompt=system_prompt,
        debug=debug,
        base_url=config.openai_compatible_url,
        api_key=config.openai_compatible_api_key,
    )


def _run_abandoning_on_cancel(fn: Callable[..., T], **kwargs: Any) -> T:
    """Run `fn` on a daemon thread and stop waiting for it once the user cancels.

    An in-flight HTTP request can't be interrupted, so it finishes (and is billed)
    in the background; its result is discarded.
    """
    raise_if_cancelled()
    future: Future = Future()
    manager = active_manager()

    def run() -> None:
        # Retries and continuations stop on this run's cancel, not a later run's state
        pin_thread(manager)
        try:
            future.set_result(fn(**kwargs))
        except BaseException as e:
            future.set_exception(e)

    threading.Thread(target=run, daemon=True).start()
    while True:
        try:
            return future.result(timeout=0.2)
        except FutureTimeoutError:
            raise_if_cancelled()


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


_NUMBERED_ITEM = re.compile(r'^\s*(\d+)\s*[:.]\s*(.*?)\s*(?=\s*\n\s*\d+\s*[:.]|\n\s*\n|\s*\Z)',
                            re.MULTILINE | re.DOTALL)


def _parse_llm_response_unified(
    response_text: str | None,
    total_elements: int,
    provider: str,
    debug: bool = False,
) -> list[str]:
    """Parse LLM response with a single numbered list.

    The item parser drops a leading and a trailing '"' from every item. That is right for an
    item the model wrapped in quotes, but narration items like '"Ah..." she said. "Hey."' lose
    their first and last quote: put them back when the item contains further quotes.
    """
    texts = _parse_llm_numbered_items(response_text, total_elements, provider, debug)
    raw = {int(n): t for n, t in _NUMBERED_ITEM.findall(_strip_code_fences(response_text or ""))}
    for i, t in enumerate(texts):
        r = raw.get(i + 1, "")
        if r.startswith('"') and r.endswith('"') and r.count('"') > 2 and t == r[1:-1].strip():
            texts[i] = r
    return texts


def _parse_llm_numbered_items(
    response_text: str | None,
    total_elements: int,
    provider: str,
    debug: bool = False,
) -> list[str]:
    """Parse LLM response with a single numbered list."""
    if response_text is None:
        log_message(f"API call failed: {provider} returned None", always_print=True)
        raise TranslationError(f"{provider}: API failed (returned None)")
    elif response_text == "":
        log_message(f"API call returned empty response: {provider}", always_print=True)
        raise TranslationError(f"{provider}: Empty response")

    try:
        log_message(
            f"Parsing {provider} unified response: {len(response_text)} chars",
            verbose=debug,
        )
        log_message(f"Raw response:\n---\n{response_text}\n---", always_print=True)

        # Pattern matches "1: text" or "1. text". An item runs until the next
        # numbered line, a blank line, or end of text (\Z, since $ would stop
        # at the first line break under MULTILINE).
        pattern = re.compile(
            r'^\s*(\d+)\s*[:.]\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*\d+\s*[:.]|\n\s*\n|\s*\Z)',
            re.MULTILINE | re.DOTALL,
        )

        matches = pattern.findall(_strip_code_fences(response_text))
        result_dict = {}

        for num_str, text in matches:
            try:
                num = int(num_str)
                if 1 <= num <= total_elements:
                    result_dict[num] = text.strip()
            except ValueError:
                continue

        final_list = []
        for i in range(1, total_elements + 1):
            if i in result_dict:
                final_list.append(result_dict[i])
            else:
                final_list.append(f"[{provider}: Missing item {i}]")

        log_message(
            f"Parsed {len(result_dict)} items from unified response (expected {total_elements})",
            verbose=debug,
        )
        return final_list

    except Exception as e:
        log_message(
            f"Failed to parse {provider} unified response: {e!s}",
            always_print=True,
        )
        return [f"[{provider}: Parse error]"] * total_elements


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


def _has_failed_items(texts: list[str], provider: str) -> bool:
    """True if any item is a failure placeholder, which a retry might fix."""
    return any(t == "[OCR FAILED]" or t.startswith(f"[{provider}") for t in texts)


def _pair_context_lines(ocr_texts: list[str], translations: list[str]) -> list[str]:
    """Join each source line to its translation so later pages reuse names and terms."""
    if len(ocr_texts) != len(translations):
        return ocr_texts
    return [
        # A leading "[" marks a failure placeholder, not a translation
        ocr
        if ocr == "[OCR FAILED]" or not tr or tr.startswith("[")
        else f"{ocr} → {tr}"
        for ocr, tr in zip(ocr_texts, translations, strict=False)
    ]


def _format_previous_context_texts(
    previous_context_texts: list[list[str]] | None,
) -> str:
    """Format previous-page OCR transcripts as a labeled context block.

    Pages are listed oldest-to-newest (matching the previous-image convention).
    Empty/failed entries are omitted, and the section is suppressed entirely
    when no usable transcripts remain.
    """
    if not previous_context_texts:
        return ""

    page_blocks = []
    for page_index, page_texts in enumerate(previous_context_texts, start=1):
        if not page_texts:
            continue
        lines = []
        for idx, text in enumerate(page_texts, start=1):
            cleaned = (text or "").strip()
            if not cleaned or cleaned == "[OCR FAILED]":
                continue
            lines.append(f"{idx}: {cleaned}")
        if not lines:
            continue
        pages_back = len(previous_context_texts) - page_index + 1
        page_blocks.append(f"### Previous page, {pages_back} back\n" + "\n".join(lines))

    if not page_blocks:
        return ""

    return (
        "\n## PREVIOUS PAGES (REFERENCE ONLY)\n"
        "Oldest first. A line with `→` gives the source text and the "
        "translation already used for it.\n" + "\n\n".join(page_blocks) + "\n"
    )


def _record_glossary_terms(
    path: str,
    pairs: list[tuple[str, str]],
    transcriptions: list[str],
    debug: bool,
) -> None:
    if not pairs:
        return
    try:
        added, rejected = record_new_terms(path, pairs, transcriptions)
    except OSError as e:
        # The translation is already paid for; losing a few terms is the lesser cost
        log_message(
            f"Warning: could not save terminology memory to {path}: {e}",
            always_print=True,
        )
        return
    if added:
        log_message(
            "Terminology memory added: "
            + ", ".join(f"{src} -> {tr}" for src, tr in added),
            always_print=True,
        )
    if rejected:
        log_message(
            "Terminology memory rejected (source not on page): "
            + ", ".join(f"{src} -> {tr}" for src, tr in rejected),
            verbose=debug,
        )


def _format_special_instructions(config: TranslationConfig) -> str:
    """Format user's special instructions section for prompts.

    Args:
        config: TranslationConfig with special_instructions

    Returns:
        Formatted special instructions string (empty if none)
    """
    if config.special_instructions and config.special_instructions.strip():
        return f"""

## CUSTOM USER INSTRUCTIONS
These come from the user and override the translation and styling rules where they conflict; the output format still applies.
{config.special_instructions.strip()}
"""
    return ""


def _perform_manga_ocr(
    images_b64: list[str],
    bubble_metadata: list[dict[str, Any]],
    debug: bool = False,
) -> list[str]:
    """Japanese OCR: manga-ocr, then Hayai re-reads and the other-language skip (_refine_ocr_reads)."""
    texts = _manga_ocr_reads(images_b64, bubble_metadata, debug)
    return _refine_ocr_reads(texts, images_b64, bubble_metadata)


def _refine_ocr_reads(
    texts: list[str], images_b64: list[str], bubble_metadata: list[dict[str, Any]]
) -> list[str]:
    """Improve the manga-ocr reads of a page.

    - Outside bubbles: Hayai OCR, line by line when comic-text-detector finds several lines
      (see core.image.line_ocr).
    - Bubbles: Hayai matches manga-ocr on regular dialogue (OpenMantra ground truth: 0.9%
      character error rate for both) but reads hand-drawn moans that manga-ocr turns into
      unrelated words ("ぅ" -> "そして、"); when one read contains the other the longer is kept.
    - Pages carrying both the Japanese text and a Chinese translation of it: the Chinese block
      is marked `mt_skip` (erased, not translated) unless MT_KEEP_OTHER_LANGUAGE is set.
    """
    import os

    from mangatl.core.image import line_ocr
    from mangatl.core.ml import hayai_ocr
    from mangatl.core.ml.model_manager import get_model_manager

    models_dir = get_model_manager().models_dir
    changed = False
    for i, meta in enumerate(bubble_metadata):
        if i >= len(texts) or not meta.get("is_outside_text"):
            continue
        text = line_ocr.read_region(meta, models_dir)
        if text:
            texts[i] = text
            changed = True

    bubble_idx = [i for i, m in enumerate(bubble_metadata)
                  if i < len(texts) and not m.get("is_outside_text")]
    if bubble_idx:
        crops = _prepare_images_for_ocr([images_b64[i] for i in bubble_idx])
        ok = [(i, c) for i, c in zip(bubble_idx, crops) if c is not None]
        reads = hayai_ocr.get(models_dir).read([c.convert("RGB") for _, c in ok],
                                               max_patches=512, max_new_tokens=160)
        for (i, _), t in zip(ok, reads):
            picked = line_ocr.pick_read(texts[i] if texts[i] != "[OCR FAILED]" else "", t.strip())
            if picked and picked != texts[i]:
                texts[i] = picked
                changed = True

    if not os.environ.get("MT_KEEP_OTHER_LANGUAGE") and any(line_ocr.is_japanese(t) for t in texts):
        for i, t in enumerate(texts):
            if i < len(bubble_metadata) and line_ocr.is_other_cjk(t):
                bubble_metadata[i]["mt_skip"] = True
                texts[i] = "(no text)"
                changed = True
                log_message(f"Skipping block {i + 1}: not in the source language "
                            "(erased, not translated)", always_print=True)
    if changed:
        log_message("Text re-read with Hayai OCR (outside bubbles: line by line via comic-text-detector):",
                    always_print=True)
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
    cache_key = cache.get_manga_ocr_cache_key(
        images_b64, total_elements, prefix="pocr16_"
    )
    cached_ocr = cache.get_manga_ocr_result(cache_key)
    if cached_ocr is not None:
        if len(cached_ocr) == total_elements:
            log_message("  - Using cached PaddleOCR-VL-1.6 results", always_print=True)
            return cached_ocr
        log_message(
            "Discarding PaddleOCR-VL-1.6 cache due to length mismatch", verbose=debug
        )

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


def _clean_previous_context_texts(
    previous_context_texts: list[list[str]] | None, max_pages: int
) -> list[list[str]]:
    """Drop empty and failed OCR lines and pages, keeping the newest `max_pages` pages.

    Filtering before trimming keeps the request order matching the prompt order
    regardless of upstream history gaps.
    """
    if not previous_context_texts or max_pages <= 0:
        return []
    cleaned = []
    for page_texts in previous_context_texts:
        usable = [
            (t or "").strip()
            for t in (page_texts or [])
            if (t or "").strip() and (t or "").strip() != "[OCR FAILED]"
        ]
        if usable:
            cleaned.append(usable)
    return cleaned[-max_pages:]


def _layout_hints(bubble_metadata: list[dict[str, Any]]) -> str:
    """Prompt line saying which items are bubble dialogue and which are outside text."""
    dialogue_indices = [
        i + 1
        for i, meta in enumerate(bubble_metadata)
        if not meta.get("is_outside_text", False)
    ]
    osb_indices = [
        i + 1
        for i, meta in enumerate(bubble_metadata)
        if meta.get("is_outside_text", False)
    ]

    hints = []
    if dialogue_indices:
        dialogue_list_str = ", ".join(map(str, dialogue_indices))
        hints.append(
            f"Items [{dialogue_list_str}] are inside speech bubbles or caption "
            "boxes (dialogue, thoughts, or narration)."
        )
    if osb_indices:
        osb_list_str = ", ".join(map(str, osb_indices))
        hints.append(
            f"Items [{osb_list_str}] are free-standing text outside bubbles "
            "(usually sound effects or mimetic effects, sometimes narration or asides)."
        )
    return "\nLayout: " + " ".join(hints) if hints else ""


def _run_ocr_step(
    config: TranslationConfig,
    images_b64: list[str],
    bubble_metadata: list[dict[str, Any]],
    debug: bool,
) -> list[str]:
    """The OCR pass with the configured local OCR method."""
    log_message("Starting OCR step", verbose=debug)
    if config.ocr_method == "manga-ocr":
        return _perform_manga_ocr(images_b64, bubble_metadata, debug)
    if config.ocr_method == "paddleocr-vl-1.6":
        return _perform_paddle_ocr_vl(images_b64, bubble_metadata, debug)
    raise TranslationError(f"Unknown OCR method: {config.ocr_method}")


def _mark_ocr_failures(
    extracted_texts: list[str], provider: str
) -> tuple[list[str], set[int]]:
    """Normalize OCR errors to "[OCR FAILED]"; returns the texts and failed indices."""
    formatted_texts = []
    ocr_failed_indices = set()
    for i, text in enumerate(extracted_texts):
        if f"[{provider}-OCR:" in text or text == "[OCR FAILED]":
            formatted_texts.append("[OCR FAILED]")
            ocr_failed_indices.add(i)
        else:
            formatted_texts.append(text)
    return formatted_texts, ocr_failed_indices


def _keep_ocr_failures(
    translations: list[str],
    ocr_failed_indices: set[int],
    total_elements: int,
    debug: bool,
) -> list[str]:
    """Force "[OCR FAILED]" on items whose OCR failed, whatever the model returned."""
    results = []
    for i in range(total_elements):
        if i not in ocr_failed_indices:
            results.append(translations[i])
            continue
        if translations[i] != "[OCR FAILED]":
            log_message(
                f"Element {i + 1}: LLM ignored OCR failure instruction",
                verbose=debug,
            )
        results.append("[OCR FAILED]")
    return results


def call_translation_api_batch(
    config: TranslationConfig,
    images_b64: list[str],
    full_image_b64: str,
    mime_types: list[str],
    full_image_mime_type: str,
    bubble_metadata: list[dict[str, Any]],
    previous_context_texts: list[list[str]] | None = None,
    ocr_texts_output: list[str] | None = None,
    debug: bool = False,
) -> list[str]:
    """
    Generates prompts and calls the appropriate LLM API endpoint based on the provider and mode
    specified in the configuration, translating text from speech bubbles and outside-bubble text.

    Supports "one-step" (OCR+Translate+Style) and "two-step" (OCR then Translate+Style) modes.

    Args:
        config (TranslationConfig): Configuration object.
        images_b64 (list): List of base64 encoded images of all text elements, in reading order.
        full_image_b64 (str): Base64 encoded image of the full manga page.
        mime_types (List[str]): List of MIME types for each text element image.
        full_image_mime_type (str): MIME type of the full page image.
        bubble_metadata (List[Dict]): List of metadata dicts with 'is_outside_text' flags for each image.
        previous_context_texts: Per-previous-page OCR transcripts (oldest-to-newest) included as
            narrative reference only. Each inner list contains the OCR strings for that page in reading order.
        ocr_texts_output: Optional mutable list. When provided, OCR transcripts (source-language text) for
            the current page's bubbles are appended in reading order so callers can propagate them as
            previous-page text context for subsequent calls.
        debug (bool): Whether to print debugging information.

    Returns:
        list: List of translated strings (potentially with style markers), one for each input text element.
              Returns placeholder messages on errors or empty responses.

    Raises:
        ValueError: If required config (API key, provider, URL) is missing or invalid.
        RuntimeError: If an API call fails irrecoverably after retries (raised by endpoint functions).
    """
    provider = config.provider
    input_language = config.input_language
    output_language = config.output_language
    cleaned_previous_texts = _clean_previous_context_texts(
        previous_context_texts,
        int(getattr(config, "previous_context_text_count", 0) or 0),
    )
    previous_context_text_count = len(cleaned_previous_texts)
    previous_text_section = _format_previous_context_texts(cleaned_previous_texts)
    glossary_path = (
        config.terminology_memory_path if config.terminology_memory else None
    )
    glossary_terms = load_terms(glossary_path) if glossary_path else {}
    previous_text_section += format_glossary_section(glossary_terms)

    total_elements = len(images_b64)
    context_hints = _layout_hints(bubble_metadata)

    cache = get_cache()
    cache_key = cache.get_translation_cache_key(
        images_b64,
        full_image_b64,
        config,
        previous_context_texts=cleaned_previous_texts,
    )
    cached_translation, cached_ocr_texts = cache.get_translation(cache_key)
    if cached_translation is not None:
        log_message("  - Using cached translation", always_print=True)
        if glossary_path:
            # Replay so a fresh glossary file grows exactly as it did on the cached run
            _record_glossary_terms(
                glossary_path,
                cache.get_translation_terms(cache_key),
                cached_ocr_texts or [],
                debug,
            )
        if ocr_texts_output is not None and cached_ocr_texts is not None:
            ocr_texts_output.extend(cached_ocr_texts)
        return cached_translation

    has_full_page = bool(config.send_full_page_context and full_image_b64)
    has_previous_context = bool(previous_context_text_count)

    try:
        extracted_texts = _run_ocr_step(config, images_b64, bubble_metadata, debug)

        log_message("Starting translation step", verbose=debug)

        formatted_texts, ocr_failed_indices = _mark_ocr_failures(
            extracted_texts, provider
        )

        ocr_input_section = """
## INPUT DATA
"""
        for i, text in enumerate(formatted_texts):
            ocr_input_section += f"{i + 1}: {text}\n"

        # Local OCR implies a text-only translation request, unless MT_PAGE_IMAGE=1
        # (--vision full: the model runs with its vision projector and sees the whole page,
        # so it can tell who speaks and catch OCR misreads)
        page_attached = has_full_page and os.environ.get("MT_PAGE_IMAGE") == "1"
        full_page_context = _FULL_PAGE_NOTE if page_attached else ""
        previous_page_context = _format_previous_context_prompt_note(
            previous_context_text_count
        )

        special_instructions_section = _format_special_instructions(config)

        translation_prompt = f"""## CONTEXT
{total_elements} transcribed text items from one comic page, in reading order.{full_page_context}{previous_page_context}
{context_hints}
{previous_text_section}
{ocr_input_section}
## TASK
Translate each item under `## INPUT DATA` into {output_language}.{special_instructions_section}
"""

        translation_parts = []
        if page_attached:
            translation_parts += [
                {"text": _FULL_PAGE_LABEL},
                _image_part(full_image_mime_type, full_image_b64),
            ]

        translation_system = _build_system_prompt_translation(
            output_language,
            full_page_context=page_attached,
            ero_doujinshi_mode=getattr(config, "ero_doujinshi_mode", False),
            ocr_correction=getattr(config, "ocr_correction", False),
            input_language=input_language,
            previous_context=has_previous_context,
            terminology_memory=bool(glossary_path),
        )
        translation_response_text = _call_llm_endpoint(
            config,
            translation_parts,
            translation_prompt,
            debug,
            system_prompt=translation_system,
        )
        new_terms = []
        if glossary_path:
            translation_response_text, new_terms = split_new_terms(
                translation_response_text
            )
            _record_glossary_terms(glossary_path, new_terms, extracted_texts, debug)
        final_translations = _parse_llm_response_unified(
            translation_response_text,
            total_elements,
            provider + "-Translate",
            debug,
        )

        if final_translations is None:
            log_message("Translation API call failed", always_print=True)
            combined_results = [
                f"[{provider}: OCR Failed]"
                if i in ocr_failed_indices
                else f"[{provider}: Translation failed]"
                for i in range(total_elements)
            ]
            if ocr_texts_output is not None:
                ocr_texts_output.extend(extracted_texts)
            return combined_results

        combined_results = _keep_ocr_failures(
            final_translations, ocr_failed_indices, total_elements, debug
        )

        if not _has_failed_items(combined_results, provider):
            cache.set_translation(
                cache_key,
                combined_results,
                ocr_texts=extracted_texts,
                new_terms=new_terms,
            )
        if ocr_texts_output is not None:
            ocr_texts_output.extend(extracted_texts)
        return combined_results
    except TranslationError:
        raise
    except (ValueError, RuntimeError) as e:
        log_message(f"Translation error: {e}", always_print=True)
        return [f"[Translation Error: {e}]"] * total_elements


def prepare_bubble_images_for_translation(
    bubble_data: list[dict[str, Any]],
    original_cv_image: np.ndarray,
    upscale_model: Any,
    device: Any,
    mime_type: str,
    bubble_min_side_pixels: int,
    upscale_method: str = "model_lite",
    whiteout_conjoined_bubbles: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """
    Prepare bubble images for translation by cropping, upscaling, color matching, and encoding.

    This function processes each speech bubble to prepare it for the translation API:
    1. Crops the bubble from the original image
    2. Upscales the bubble to meet minimum size requirements (based on upscale_method)
    3. Matches colors to preserve visual consistency (only for model upscaling)
    4. Encodes the processed bubble as base64 for API transmission

    Args:
        bubble_data: List of bubble detection dicts with 'bbox' keys
        original_cv_image: OpenCV image array of the original image
        upscale_model: Loaded upscaling model
        device: PyTorch device for model inference
        mime_type: MIME type for image encoding
        upscale_method: Method for upscaling - "model", "lanczos", or "none"
        verbose: Whether to print detailed logging

    Returns:
        List of bubble dicts with added 'image_b64' and 'mime_type' keys
        (immutable approach - returns new list without mutating input)
    """
    cv2_ext = ".png" if mime_type == "image/png" else ".jpg"

    prepared_bubbles = []

    mask_lookup = {}
    for b in bubble_data:
        b_bbox = tuple(round(v) for v in b["bbox"])
        mask_lookup[b_bbox] = b.get("sam_mask")

    if upscale_method == "model":
        log_message(
            f"Upscaling {len(bubble_data)} bubble images with 2x-AnimeSharpV4_RCAN",
            always_print=True,
        )
    elif upscale_method == "model_lite":
        log_message(
            f"Upscaling {len(bubble_data)} bubble images with 2x-AnimeSharpV4_Fast_RCAN_PU (Lite)",
            always_print=True,
        )
    elif upscale_method == "lanczos":
        log_message(
            f"Upscaling {len(bubble_data)} bubble images with LANCZOS",
            always_print=True,
        )
    else:
        log_message(
            f"Processing {len(bubble_data)} bubble images without upscaling",
            always_print=True,
        )

    for bubble in bubble_data:
        prepared_bubble = bubble.copy()
        bx1, by1, bx2, by2 = [round(c) for c in bubble["bbox"]]
        bx1 = max(0, min(original_cv_image.shape[1], bx1))
        by1 = max(0, min(original_cv_image.shape[0], by1))
        bx2 = max(0, min(original_cv_image.shape[1], bx2))
        by2 = max(0, min(original_cv_image.shape[0], by2))
        if bx2 > bx1 and by2 > by1:
            prepared_bubble["original_crop_pil"] = cv2_to_pil(
                original_cv_image[by1:by2, bx1:bx2].copy()
            )

        x1, y1, x2, y2 = (max(0, round(v)) for v in bubble["bbox"])

        # Use the tight bbox of the mask
        _mask = bubble.get("sam_mask")
        _ma = None
        if _mask is not None:
            _ma = np.asarray(_mask)
            if _ma.ndim == 3:
                _ma = _ma[..., 0]
            if _ma.ndim == 2:
                mx, my, mw, mh = cv2.boundingRect((_ma > 0).view(np.uint8))
                if mw and mh:
                    x1 = min(x1, mx)
                    y1 = min(y1, my)
                    x2 = max(x2, mx + mw)
                    y2 = max(y2, my + mh)

        bubble_image_cv = original_cv_image[y1:y2, x1:x2].copy()

        # White-out conjoined neighbor text regions visible in this crop
        neighbor_bboxes = bubble.get("conjoined_neighbor_bboxes")
        if whiteout_conjoined_bubbles and neighbor_bboxes:
            own_mask_crop = (
                _ma[y1:y2, x1:x2] > 0 if (_ma is not None and _ma.ndim == 2) else None
            )

            for nb in neighbor_bboxes:
                nb_tuple = tuple(round(v) for v in nb)
                neighbor_mask = mask_lookup.get(nb_tuple)

                if neighbor_mask is not None:
                    _nm = np.asarray(neighbor_mask)
                    if _nm.ndim == 3:
                        _nm = _nm[..., 0]
                    if _nm.ndim == 2:
                        nm_crop = _nm[y1:y2, x1:x2] > 0

                        if own_mask_crop is not None:
                            region_mask = nm_crop & ~own_mask_crop
                        else:
                            region_mask = nm_crop

                        # Apply whiteout precisely on neighbor's mask pixels
                        bubble_image_cv[region_mask] = 255

        bubble_image_pil = cv2_to_pil(bubble_image_cv)

        if upscale_method in {"model", "model_lite"}:
            final_bubble_pil = process_bubble_image_cached(
                bubble_image_pil,
                upscale_model,
                device,
                bubble_min_side_pixels,
                "min",
                upscale_method,
                verbose,
            )
        elif upscale_method == "lanczos":
            w, h = bubble_image_pil.size
            min_side = min(w, h)
            if min_side < bubble_min_side_pixels:
                scale_factor = bubble_min_side_pixels / min_side
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                resized_bubble = resize_lanczos_antiring(
                    bubble_image_pil, (new_w, new_h)
                )
            else:
                resized_bubble = bubble_image_pil
            final_bubble_pil = resized_bubble
        else:
            final_bubble_pil = bubble_image_pil

        final_bubble_cv = pil_to_cv2(final_bubble_pil)

        try:
            is_success, buffer = cv2.imencode(cv2_ext, final_bubble_cv)
            if is_success:
                image_b64 = base64.b64encode(buffer).decode("utf-8")
                prepared_bubble["image_b64"] = image_b64
                prepared_bubble["mime_type"] = mime_type
                log_message(
                    f"Bubble {x1},{y1} ({final_bubble_pil.size[0]}x{final_bubble_pil.size[1]})",
                    verbose=verbose,
                )
            else:
                log_message(
                    f"Failed to encode bubble {bubble['bbox']}", verbose=verbose
                )
                prepared_bubble["image_b64"] = None
        except Exception as e:
            log_message(f"Error encoding bubble {bubble['bbox']}: {e}", verbose=verbose)
            prepared_bubble["image_b64"] = None

        prepared_bubbles.append(prepared_bubble)

    return prepared_bubbles
