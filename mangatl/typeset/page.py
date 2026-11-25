"""Typesetting a page: font sizes from the source lettering, then every translation drawn into its bubble or onto the art."""

import dataclasses
import math
import re
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import cv2
import numpy as np
from PIL import Image

from mangatl.clean.bubbles import inpaints_colored_bubbles, retry_cleaning_with_otsu
from mangatl.clean.outside_text import paint_pending_osb_fill, paint_pending_osb_fills
from mangatl.common.cancellation import raise_if_manager_cancelled
from mangatl.common.exceptions import FontError, ImageProcessingError, RenderingError
from mangatl.common.image_utils import estimate_glyph_size
from mangatl.common.log import log_message
from mangatl.common.scaling import scale_font_size, scale_scalar
from mangatl.detect.tilt import compute_effective_tilt_angle
from mangatl.ocr.line_ocr import script_ratios
from mangatl.settings import PROVIDER, Settings
from mangatl.typeset import sfx_labels
from mangatl.typeset.fonts import find_font_variants, get_cap_height_ratio
from mangatl.typeset.renderer import RenderingConfig, render_text_skia
from mangatl.typeset.text import supports_long_word_breaking

if TYPE_CHECKING:
    from mangatl.common.cancellation import CancellationManager

INVALID_TRANSLATION_VALUES = {"[OCR FAILED]", "[Empty response / no content]"}


# Translated cap height relative to the measured original lettering. CJK glyphs fill the em
# box and their translations run longer, so professional lettering sits near 0.71 of it.
# Other scripts are measured at about cap height already, so they are matched as is.
AUTO_FONT_CJK_GLYPH_RATIO = 0.71


AUTO_FONT_CJK_LANGUAGE_PREFIXES = ("japanese", "korean", "chinese")


# Typical translated cap height on a 1 MP page, assumed when a page has no measurable text
AUTO_FONT_TYPICAL_CAP_HEIGHT = 13.0


# Ink parts at which one bubble's measurement is fully trusted. Bubbles with fewer count for
# less, and a page that cannot reach it in total (e.g. a lone "..." bubble) is unmeasurable.
AUTO_FONT_TRUSTED_INK_PARTS = 10


def _page_glyph_size(
    page_image: np.ndarray, bubbles_info: list[dict[str, Any]], full_width: bool
) -> float | None:
    """Typical original lettering size across the page's speech bubbles, None if unmeasurable."""
    gray = page_image if page_image.ndim == 2 else cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    measured = []
    for info in bubbles_info:
        mask = info.get("mask")
        if mask is None:
            continue
        size_and_parts = estimate_glyph_size(gray, info["bbox"], mask, full_width)
        if size_and_parts:
            size, parts = size_and_parts
            measured.append((size, min(parts, AUTO_FONT_TRUSTED_INK_PARTS)))
    total_weight = sum(weight for _, weight in measured)
    if total_weight < AUTO_FONT_TRUSTED_INK_PARTS:
        return None

    # Weighted median, so ellipses and shouts in a few giant strokes are outvoted
    measured.sort()
    weight_so_far = 0
    for i, (size, weight) in enumerate(measured):
        weight_so_far += weight
        if weight_so_far * 2 > total_weight:
            return size
        if weight_so_far * 2 == total_weight:
            return (size + measured[i + 1][0]) / 2


def _bubble_glyph_sizes(
    page_image: np.ndarray, bubbles_info: list[dict[str, Any]], full_width: bool
) -> dict[tuple, tuple]:
    """Each speech bubble's own lettering size (estimate_glyph_size) by its bbox."""
    gray = page_image if page_image.ndim == 2 else cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    own = {}
    for info in bubbles_info:
        if info.get("mask") is None or not info.get("bbox"):
            continue
        est = estimate_glyph_size(gray, info["bbox"], info["mask"], full_width)
        if est:
            own[tuple(int(v) for v in info["bbox"])] = est
    return own


def resolve_main_max_font(
    config: Settings,
    page_image: np.ndarray,
    bubbles_info: list[dict[str, Any]],
    min_font: int,
    max_font: int,
    processing_scale: float,
    strip_glyph_size: list[float] | None,
    verbose: bool,
    page_glyphs: dict | None = None,
) -> int:
    """Speech bubble max font size, lowered to the auto size when that is enabled.

    When the page's lettering is measured here, `page_glyphs` receives the page's typical size
    ("page") and each bubble's own (glyph size, ink parts) by bbox ("own"), for _emphasis_sizes.
    Sections of a long strip that reuse the first section's size get no emphasis data.
    """
    if not config.typesetting.bubbles.auto_font_size:
        return max_font

    cjk_source = config.translation.source_language.lower().startswith(AUTO_FONT_CJK_LANGUAGE_PREFIXES)
    if strip_glyph_size:
        glyph_size = strip_glyph_size[0]
    else:
        glyph_size = _page_glyph_size(page_image, bubbles_info, cjk_source)
        if glyph_size is not None and strip_glyph_size is not None:
            strip_glyph_size.append(glyph_size)
        if page_glyphs is not None:
            page_glyphs["page"] = glyph_size
            page_glyphs["own"] = _bubble_glyph_sizes(page_image, bubbles_info, cjk_source)

    if glyph_size is None:
        cap_height = AUTO_FONT_TYPICAL_CAP_HEIGHT * processing_scale
    else:
        cap_height = glyph_size * (AUTO_FONT_CJK_GLYPH_RATIO if cjk_source else 1.0)
    regular_font_path = find_font_variants(config.typesetting.bubbles.font_dir, verbose=verbose)["regular"]
    auto_max = round(cap_height / get_cap_height_ratio(str(regular_font_path)))
    auto_max = max(min_font, min(max_font, auto_max))
    log_message(f"Auto font size: {auto_max}", verbose=verbose)
    return auto_max


class _RenderSizes(NamedTuple):
    """Font sizes and paddings scaled to the page."""

    main_min_font: int
    main_max_font: int
    osb_min_font: int
    osb_max_font: int
    padding_pixels: float
    osb_padding_pixels: float
    osb_outline_ratio: float


def scaled_render_sizes(config: Settings, processing_scale: float) -> _RenderSizes:
    main_min_font = scale_font_size(
        config.typesetting.bubbles.min_font_size, processing_scale, minimum=4, maximum=256
    )
    osb_min_font = scale_font_size(
        config.typesetting.outside.min_font_size,
        processing_scale,
        minimum=4,
        maximum=512,
    )
    return _RenderSizes(
        main_min_font=main_min_font,
        main_max_font=scale_font_size(
            config.typesetting.bubbles.max_font_size,
            processing_scale,
            minimum=main_min_font,
            maximum=384,
        ),
        osb_min_font=osb_min_font,
        osb_max_font=scale_font_size(
            config.typesetting.outside.max_font_size,
            processing_scale,
            minimum=osb_min_font,
            maximum=640,
        ),
        padding_pixels=scale_scalar(
            config.typesetting.bubbles.padding,
            processing_scale,
            minimum=1.0,
            maximum=80.0,
        ),
        osb_padding_pixels=scale_scalar(
            config.typesetting.outside.padding,
            processing_scale,
            minimum=1.0,
            maximum=80.0,
        ),
        osb_outline_ratio=config.typesetting.outside.outline_ratio,
    )


def _paste_original_crop(image: Image.Image, bubble: dict[str, Any]) -> Image.Image:
    bbox = bubble["bbox"]
    restored = image.copy()
    restored.paste(bubble["original_crop_pil"], (round(bbox[0]), round(bbox[1])))
    return restored


def _text_render_config(config: Settings, sizes: _RenderSizes, is_outside_text: bool) -> RenderingConfig:
    ts = config.typesetting
    style = ts.outside if is_outside_text else ts.bubbles
    # Hyphenating languages insert hyphens; Korean/Thai use
    # no-hyphen emergency breaks under the same user setting.
    should_hyphenate = ts.hyphenate_before_scaling and supports_long_word_breaking(
        config.translation.target_language
    )
    return RenderingConfig(
        min_font_size=sizes.osb_min_font if is_outside_text else sizes.main_min_font,
        max_font_size=sizes.osb_max_font if is_outside_text else sizes.main_max_font,
        line_spacing_mult=style.line_spacing,
        use_ligatures=style.ligatures,
        hyphenate_before_scaling=should_hyphenate,
        hyphenation_min_word_length=ts.hyphenation_min_word_length,
        hyphenation_language=config.translation.target_language,
        badness_exponent=ts.badness_exponent,
        padding_pixels=(sizes.osb_padding_pixels if is_outside_text else sizes.padding_pixels),
        outline_ratio=sizes.osb_outline_ratio if is_outside_text else 0.0,
        supersampling_factor=ts.supersampling,
        detach_trailing_punctuation=ts.detach_trailing_punctuation,
        auto_vertical_text=style.auto_vertical,
        vertical_line_spacing_mult=style.vertical_line_spacing,
        vertical_font_size_mult=style.vertical_font_size,
    )


_OsbAttempt = tuple[float, bool, tuple[float, float] | None]


def _osb_render_attempts(bubble: dict[str, Any], config: Settings, sizes: _RenderSizes) -> list[_OsbAttempt]:
    """(rotation, vertical stack, layout dimensions) to try for an outside text, in order."""
    if bubble.get("mt_narration"):  # narration / dialogue on the art: upright (typeset.narration)
        return [(0.0, False, None), (0.0, True, None)]
    verbose = config.verbose
    osb = config.typesetting.outside
    bbox = bubble["bbox"]
    rotation_deg = 0.0
    if osb.follow_tilt:
        rotation_deg = compute_effective_tilt_angle(
            tilt_deg=float(bubble.get("tilt_deg", 0.0) or 0.0),
            tilt_conf=float(bubble.get("tilt_conf", 0.0) or 0.0),
            follow_tilt=osb.follow_tilt,
            confidence_threshold=osb.tilt_confidence,
            min_tilt_deg=osb.min_tilt,
            max_tilt_deg=osb.max_tilt,
        )
        if abs(rotation_deg) > 0.01:
            conf = float(bubble.get("tilt_conf", 0.0) or 0.0)
            log_message(
                f"OSB tilt applied: {rotation_deg:.1f} deg (conf: {conf:.2f}, axis: {bubble.get('tilt_axis', 'none')})",
                verbose=verbose,
            )

    tilted_layout_dims = None
    if abs(rotation_deg) > 0.01:
        shrink = max(0.70, math.cos(math.radians(rotation_deg)))
        avail_w = max(1.0, (bbox[2] - bbox[0]) - 2 * sizes.osb_padding_pixels)
        avail_h = max(1.0, (bbox[3] - bbox[1]) - 2 * sizes.osb_padding_pixels)
        tilted_layout_dims = (avail_w * shrink, avail_h * shrink)

    # OSB renders default to horizontal; vertical stacking is fallback-only. Try the
    # detected tilt first, then vertical, then upright in each orientation.
    attempts = [(rotation_deg, False, tilted_layout_dims)]
    if abs(rotation_deg) > 0.01:
        attempts += [(rotation_deg, True, tilted_layout_dims), (0.0, False, None)]
    attempts.append((0.0, True, None))
    return attempts


def _render_osb_attempt(
    image: Image.Image,
    bubble: dict[str, Any],
    text: str,
    bubble_id: str,
    config: Settings,
    sizes: _RenderSizes,
    attempt: _OsbAttempt,
    layout_only: bool = False,
) -> Image.Image:
    """Render outside text with one attempt's settings; raises if it doesn't fit."""
    if bubble.get("mt_narration"):
        # narration / dialogue on the art: normal case, one utterance per line, dialogue font
        # (typeset.narration); the caller passes text.upper()
        text = bubble.get("mt_text", text)
        config = config.updated(**{"typesetting.outside.font": config.typesetting.bubbles.font})
    rotation_deg, vertical_stack, layout_dims = attempt
    is_dark_text = bubble.get("is_dark_text", True)
    text_color_rgb = bubble.get("text_color_rgb", None)
    text_bg_rgb = None
    if bubble.get("needs_text_background"):
        if text_color_rgb:
            lum = 0.299 * text_color_rgb[0] + 0.587 * text_color_rgb[1] + 0.114 * text_color_rgb[2]
            text_bg_rgb = (255, 255, 255) if lum < 128 else (0, 0, 0)
        else:
            text_bg_rgb = (0, 0, 0) if is_dark_text else (255, 255, 255)

    return render_text_skia(
        pil_image=image,
        text=text,
        bbox=bubble["bbox"],
        font_dir=config.typesetting.outside.font_dir or config.typesetting.bubbles.font_dir,
        # Outside text was inpainted, no mask needed
        cleaned_mask=None,
        bubble_color_bgr=(50, 50, 50) if is_dark_text else (255, 255, 255),
        config=_text_render_config(config, sizes, is_outside_text=True),
        verbose=config.verbose,
        bubble_id=bubble_id,
        rotation_deg=rotation_deg,
        vertical_stack=vertical_stack,
        text_color_rgb=text_color_rgb,
        raise_on_safe_error=False,
        text_background_color=text_bg_rgb,
        fallback_padding_pixels=sizes.osb_padding_pixels,
        layout_dimensions=layout_dims,
        layout_only=layout_only,
    )


def _fitting_osb_attempts(
    image: Image.Image,
    bubble: dict[str, Any],
    text: str,
    bubble_id: str,
    config: Settings,
    sizes: _RenderSizes,
) -> list[_OsbAttempt]:
    """The OSB attempts from the first one whose layout fits; empty if none fits."""
    attempts = _osb_render_attempts(bubble, config, sizes)
    for k, attempt in enumerate(attempts):
        try:
            _render_osb_attempt(image, bubble, text, bubble_id, config, sizes, attempt, True)
        except Exception as e:
            log_message(
                f"OSB layout failed at {attempt[0]:.1f} deg (vertical_stack={attempt[1]}): {e}",
                verbose=config.verbose,
            )
            continue
        return attempts[k:]
    return []


def _render_outside_text(
    image: Image.Image,
    bubble: dict[str, Any],
    text: str,
    bubble_id: str,
    config: Settings,
    sizes: _RenderSizes,
    attempts: list[_OsbAttempt],
) -> Image.Image | None:
    """Render one outside text translation, or None if it could not be placed."""
    verbose = config.verbose
    bbox = bubble["bbox"]
    log_message(f"Rendering outside text {bbox}: '{text[:30]}...'", verbose=verbose)
    for attempt in attempts:
        try:
            return _render_osb_attempt(image, bubble, text, bubble_id, config, sizes, attempt)
        except Exception as e:
            log_message(
                f"OSB render failed at {attempt[0]:.1f} deg (vertical_stack={attempt[1]}): {e}",
                verbose=verbose,
            )

    # Last resort: restore the original OSB patch if available
    if "original_crop_pil" in bubble:
        log_message(
            f"Restoring original OSB patch for {bbox}",
            verbose=verbose,
            always_print=True,
        )
        return _paste_original_crop(image, bubble)
    return None


EMPHASIS_MIN_RATIO = 1.4  # source lettering this much above the page's typical size is emphasis


EMPHASIS_MAX_SCALE = 3.0  # at most this much bigger than the page's dialogue size


def _emphasis_sizes(
    bubble: dict[str, Any],
    bubble_id: str,
    sizes: _RenderSizes,
    page_glyphs: dict | None,
    config: Settings,
) -> _RenderSizes:
    """Every bubble is capped at one page-wide size taken from the typical source glyph, so a
    shout lettered three times bigger would get the same small English as a whisper, lost in
    its big bubble. Professional letterers follow the source: bubbles with clearly bigger
    lettering (page_glyphs, measured by resolve_main_max_font) grow by the same ratio."""
    page = (page_glyphs or {}).get("page")
    est = _lookup_bubble_glyph(page_glyphs["own"], bubble.get("bbox", ())) if page else None
    # A lone "..." or "!!" is a few strokes; trust only measurements from enough ink
    if est and est[1] >= 3 and est[0] >= EMPHASIS_MIN_RATIO * page:
        # a letterer exaggerates a little: shouts read bigger than their glyph ratio
        scale = min((est[0] / page) ** 1.5, EMPHASIS_MAX_SCALE)
        bigger = round(sizes.main_max_font * scale)
        log_message(
            f"Bubble {bubble_id}: source lettering {est[0] / page:.1f}x the page's, "
            f"max font {sizes.main_max_font} -> {bigger}",
            verbose=config.verbose,
        )
        sizes = sizes._replace(main_max_font=bigger)
    return sizes


def _lookup_bubble_glyph(own: dict, bbox) -> tuple | None:
    key = tuple(int(v) for v in bbox)
    if key in own:
        return own[key]
    best, best_d = None, 1e9
    for k, v in own.items():
        d = sum(abs(a - b) for a, b in zip(k, key))
        if d < best_d:
            best, best_d = v, d
    return best if best_d <= 8 else None


def _render_bubble(
    image: Image.Image,
    bubble: dict[str, Any],
    text: str,
    bubble_id: str,
    render_info: dict[str, Any] | None,
    page_cv: np.ndarray,
    config: Settings,
    sizes: _RenderSizes,
    processing_scale: float,
    page_glyphs: dict | None = None,
) -> Image.Image | None:
    """Render one speech bubble translation, sized to its own lettering (_emphasis_sizes).

    A translation that does not fit at the minimum font size (tall, narrow bubbles, often after
    an OCR misread) is retried at half the minimum size; if it still does not fit, the original
    lettering goes back instead of leaving the bubble blank. Colored / see-through bubbles
    (outline-only bubbles over the art, common in doujin CG) are inpainted, not filled, so
    their text gets an outline like text on the art.
    """
    sizes = _emphasis_sizes(bubble, bubble_id, sizes, page_glyphs, config)
    outlined = bool(render_info and render_info.get("is_colored"))
    rendered = _render_bubble_once(
        image, bubble, text, bubble_id, render_info, page_cv, config, sizes, processing_scale, outlined
    )
    if rendered is not None:
        return rendered
    smaller = sizes._replace(main_min_font=max(4, sizes.main_min_font // 2))
    rendered = _render_bubble_once(
        image, bubble, text, bubble_id, render_info, page_cv, config, smaller, processing_scale, outlined
    )
    if rendered is not None:
        log_message(f"Bubble {bubble_id}: fitted below the minimum font size", always_print=True)
        return rendered
    if "original_crop_pil" in bubble:
        log_message(
            f"Bubble {bubble_id}: translation does not fit, kept the original lettering ('{text[:30]}')",
            always_print=True,
        )
        return _paste_original_crop(image, bubble)
    return None


def _render_bubble_once(
    image: Image.Image,
    bubble: dict[str, Any],
    text: str,
    bubble_id: str,
    render_info: dict[str, Any] | None,
    page_cv: np.ndarray,
    config: Settings,
    sizes: _RenderSizes,
    processing_scale: float,
    outlined: bool = False,
) -> Image.Image | None:
    """Render one speech bubble translation, or None if it could not be placed.

    When the cleaning mask leaves no safe area, the mask is rebuilt with Otsu and the
    render retried, then a padded bbox render is the final fallback.
    """
    verbose = config.verbose
    bbox = bubble["bbox"]
    log_message(f"Rendering bubble {bbox}: '{text[:30]}...'", verbose=verbose)

    bubble_color_bgr = (255, 255, 255)
    cleaned_mask = None
    base_mask = None
    is_seg_mask = False
    text_color_rgb = None
    if render_info:
        bubble_color_bgr = render_info["color"]
        cleaned_mask = render_info.get("mask")
        base_mask = render_info.get("base_mask")
        is_seg_mask = render_info.get("is_seg", False)
        text_color_bgr = render_info.get("text_color_bgr")
        if text_color_bgr:
            text_color_rgb = (text_color_bgr[2], text_color_bgr[1], text_color_bgr[0])

    render_config = _text_render_config(config, sizes, is_outside_text=False)
    if outlined:
        render_config = dataclasses.replace(render_config, outline_ratio=sizes.osb_outline_ratio)

    def render(mask, color, raise_on_safe_error: bool) -> Image.Image:
        return render_text_skia(
            pil_image=image,
            text=text,
            bbox=bbox,
            font_dir=config.typesetting.bubbles.font_dir,
            cleaned_mask=mask,
            bubble_color_bgr=color,
            config=render_config,
            verbose=verbose,
            bubble_id=bubble_id,
            rotation_deg=0.0,
            vertical_stack=False,
            text_color_rgb=text_color_rgb,
            raise_on_safe_error=raise_on_safe_error,
        )

    try:
        return render(cleaned_mask, bubble_color_bgr, raise_on_safe_error=True)
    except ImageProcessingError as e:
        safe_area_failed = "Safe area calculation failed" in str(e)
        retry_result = None
        if safe_area_failed and base_mask is not None:
            log_message(
                f"Safe area failed for bubble {bbox}, retrying mask with Otsu",
                verbose=verbose,
                always_print=True,
            )
            retry_result = retry_cleaning_with_otsu(
                page_cv,
                {
                    "base_mask": base_mask,
                    "bbox": bbox,
                    "is_seg": is_seg_mask,
                    "is_colored": (render_info.get("is_colored", False) if render_info else False),
                    "text_bbox": render_info.get("text_bbox") if render_info else None,
                    "text_color_bgr": (render_info.get("text_color_bgr") if render_info else None),
                },
                config.cleaning.threshold,
                config.cleaning.roi_shrink_px,
                processing_scale,
                verbose=verbose,
                classify_colored=inpaints_colored_bubbles(config),
            )

        if retry_result and retry_result.get("mask") is not None:
            cleaned_mask = retry_result["mask"]
            bubble_color_bgr = retry_result.get("color", bubble_color_bgr)
            base_mask = retry_result.get("base_mask", base_mask)
            if render_info is not None:
                render_info.update(
                    {
                        "mask": cleaned_mask,
                        "color": bubble_color_bgr,
                        "base_mask": base_mask,
                        "is_colored": retry_result.get("is_colored", render_info.get("is_colored", False)),
                        "text_bbox": retry_result.get("text_bbox", render_info.get("text_bbox")),
                    }
                )
            try:
                return render(cleaned_mask, bubble_color_bgr, raise_on_safe_error=False)
            except (RenderingError, FontError, ImageProcessingError) as e2:
                log_message(f"Text rendering failed after Otsu retry: {e2}", verbose=verbose)

        log_message(
            f"Safe area calculation failed for {bbox}, using padded bbox fallback"
            if safe_area_failed
            else f"Rendering retry fallback for {bbox}, using padded bbox method",
            verbose=verbose,
        )
        try:
            return render(cleaned_mask, bubble_color_bgr, raise_on_safe_error=False)
        except (RenderingError, FontError) as e2:
            log_message(f"Text rendering failed: {e2}", verbose=verbose)
            return None
    except (RenderingError, FontError) as e:
        log_message(f"Text rendering failed: {e}", verbose=verbose)
        return None


def _render_action(bubble: dict[str, Any], text: str, provider_tag: str) -> str:
    """_basic_render_action, plus "keep_original" for outside text whose OCR read is junk
    (speed lines or artwork the detector mistook for lettering) and for sound effects that get
    a label beside the artist's lettering (typeset.sfx_labels)."""
    action = _basic_render_action(bubble, text, provider_tag)
    if (
        action == "render"
        and bubble.get("is_outside_text")
        and "original_crop_pil" in bubble
        and _junk_read(bubble.get("ocr_text", ""))
    ):
        action = "keep_original"
    if bubble.get("_sfx_label") and action not in ("ocr_failed", "invalid"):  # label mode only
        action = "keep_original"
    return action


def _junk_read(ocr: str) -> bool:
    """An outside-text "read" with no Japanese and at most 3 letters is speed lines or
    artwork the detector mistook for lettering ("H")."""
    t = re.sub(r"^\[(OSB|Bubble)\]\s*", "", ocr or "").strip()
    if not t or t.startswith("["):
        return False
    _, kana, han = script_ratios(t)
    return kana + han == 0 and sum(c.isalnum() for c in t) <= 3 and not set(t) <= set("!?.…")


def _basic_render_action(bubble: dict[str, Any], text: str, provider_tag: str) -> str:
    """How to handle one translation: "ocr_failed", "invalid", "keep_original"
    (outside text the translation left unchanged) or "render"."""
    ocr_text = (bubble.get("ocr_text") or "").strip()
    trimmed_text = (text or "").strip()
    if (
        trimmed_text.upper() == "[OCR FAILED]"
        or ocr_text.upper() == "[OCR FAILED]"
        or ": OCR FAILED]" in trimmed_text.upper()
    ):
        return "ocr_failed"
    if (
        not text
        or text.startswith(("API Error", "[Translation Error]", "[Translation Error:", provider_tag))
        or trimmed_text in INVALID_TRANSLATION_VALUES
    ):
        return "invalid"
    if (
        bubble.get("is_outside_text", False)
        and ocr_text
        and ocr_text not in INVALID_TRANSLATION_VALUES
        and ocr_text == trimmed_text
        and "original_crop_pil" in bubble
    ):
        return "keep_original"
    return "render"


def _paint_osb_fills(
    image: Image.Image,
    sorted_bubble_data: list[dict[str, Any]],
    translated_texts: list[str],
    provider_tag: str,
    config: Settings,
    sizes: _RenderSizes,
) -> tuple[Image.Image, dict[int, list[_OsbAttempt]]]:
    """Paint each outside text's pending fill at the angle its translation fits at.

    All fills go down before any text, since a tilted fill can reach past its box.
    Returns the page and, per rendered outside text index, the attempts to draw it with.
    """
    attempts_by_index = {}
    for i, (bubble, text) in enumerate(zip(sorted_bubble_data, translated_texts, strict=True)):
        if not bubble.get("is_outside_text", False):
            continue
        action = _render_action(bubble, text, provider_tag)
        attempts = []
        if action == "render":
            attempts = _fitting_osb_attempts(image, bubble, text.upper(), str(i + 1), config, sizes)
            attempts_by_index[i] = attempts
            if attempts:
                image = paint_pending_osb_fill(image, bubble, attempts[0][0])
                continue
        if action != "invalid" and "original_crop_pil" in bubble:
            bubble.pop("pending_fill", None)  # the original patch goes back
        else:
            image = paint_pending_osb_fill(image, bubble)
    return image, attempts_by_index


def render_translations(
    image: Image.Image,
    sorted_bubble_data: list[dict[str, Any]],
    translated_texts: list[str],
    processed_bubbles_info: list[dict[str, Any]],
    page_cv: np.ndarray,
    config: Settings,
    sizes: _RenderSizes,
    processing_scale: float,
    empty_bubbles: list,
    cancellation_manager: Optional["CancellationManager"],
    page_glyphs: dict | None = None,
) -> Image.Image:
    """Render the page's translations (_render_page_texts); with --sfx label, sound effects keep
    their lettering and get an English label beside it (typeset.sfx_labels)."""
    labels = (
        sfx_labels.mark_labels(sorted_bubble_data, translated_texts)
        if config.typesetting.sfx == "label"
        else []
    )
    image = _render_page_texts(
        image,
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
    return sfx_labels.draw_labels(image, labels, sorted_bubble_data, config, sizes)


def _render_page_texts(
    image: Image.Image,
    sorted_bubble_data: list[dict[str, Any]],
    translated_texts: list[str],
    processed_bubbles_info: list[dict[str, Any]],
    page_cv: np.ndarray,
    config: Settings,
    sizes: _RenderSizes,
    processing_scale: float,
    empty_bubbles: list,
    cancellation_manager: Optional["CancellationManager"],
    page_glyphs: dict | None = None,
) -> Image.Image:
    """Render each translation onto the cleaned page, recording bubbles left empty."""
    verbose = config.verbose
    render_info_by_bbox = {
        tuple(info["bbox"]): {
            "color": info["color"],
            "mask": info.get("mask"),
            "base_mask": info.get("base_mask"),
            "is_seg": info.get("is_seg", False),
            "is_colored": info.get("is_colored", False),
            "text_bbox": info.get("text_bbox"),
            "text_color_bgr": info.get("text_color_bgr"),
        }
        for info in processed_bubbles_info
        if "bbox" in info and "color" in info and "mask" in info
    }
    log_message("Rendering translations...", verbose=verbose)
    if len(translated_texts) != len(sorted_bubble_data):
        log_message(
            f"Warning: Bubble/translation count mismatch ({len(sorted_bubble_data)}/{len(translated_texts)})",
            always_print=True,
        )
        # blocks erased on purpose (mt_skip: other-language text) are not "left empty"
        empty_bubbles.extend(b["bbox"] for b in sorted_bubble_data if not b.get("mt_skip"))
        return paint_pending_osb_fills(image, sorted_bubble_data)

    provider_tag = f"[{PROVIDER}:"
    image, osb_attempts = _paint_osb_fills(
        image, sorted_bubble_data, translated_texts, provider_tag, config, sizes
    )
    for i, (bubble, text) in enumerate(zip(sorted_bubble_data, translated_texts, strict=True)):
        raise_if_manager_cancelled(cancellation_manager)
        bubble["translation"] = text
        bbox = bubble["bbox"]
        is_outside_text = bubble.get("is_outside_text", False)
        entry_type = "outside text" if is_outside_text else "bubble"
        action = _render_action(bubble, text, provider_tag)

        if action == "ocr_failed":
            if "original_crop_pil" in bubble:
                log_message(
                    f"Restoring original {entry_type} patch due to OCR failure for {bbox}",
                    verbose=verbose,
                    always_print=True,
                )
                image = _paste_original_crop(image, bubble)
            else:
                log_message(
                    f"Skipping {entry_type} {bbox} - OCR failed and no original crop available",
                    verbose=verbose,
                )
                if not bubble.get("mt_skip"):
                    empty_bubbles.append(bbox)
            continue

        if action == "invalid":
            log_message(f"Skipping {entry_type} {bbox} - invalid translation", verbose=verbose)
            if not bubble.get("mt_skip"):  # erased on purpose (other-language block)
                empty_bubbles.append(bbox)
            continue

        if action == "keep_original":
            log_message(
                f"Restoring original OSB patch because OCR matches translation for {bbox}",
                verbose=verbose,
                always_print=True,
            )
            image = _paste_original_crop(image, bubble)
            continue

        if is_outside_text:
            text = text.upper()
            bubble["translation"] = text
            rendered = _render_outside_text(image, bubble, text, str(i + 1), config, sizes, osb_attempts[i])
        else:
            rendered = _render_bubble(
                image,
                bubble,
                text,
                str(i + 1),
                render_info_by_bbox.get(tuple(bbox)),
                page_cv,
                config,
                sizes,
                processing_scale,
                page_glyphs,
            )

        if rendered is None:
            log_message(f"Failed to render bubble {bbox}", verbose=verbose)
            if not bubble.get("mt_skip"):
                empty_bubbles.append(bbox)
        else:
            image = rendered
    return image
