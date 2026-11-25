from pathlib import Path
from typing import Any


SETTING_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "image_upscale_factor": (1.0, 8.0),
    "confidence": (0.1, 1.0),
    "conjoined_confidence": (0.1, 1.0),
    "panel_confidence": (0.05, 1.0),
    "thresholding_value": (0, 255),
    "roi_shrink_px": (0, 10),
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (0, 64),
    "max_tokens": (4096, 63488),
    "bubble_min_side_pixels": (64, 512),
    "context_image_max_side_pixels": (512, 2560),
    "previous_context_text_count": (0, 50),
    "batch_previous_context_text_count": (0, 50),
    "osb_min_side_pixels": (64, 512),
    "max_font_size": (5, 50),
    "min_font_size": (5, 50),
    "line_spacing_mult": (0.5, 2.0),
    "vertical_line_spacing_mult": (0.5, 2.0),
    "vertical_font_size_mult": (0.1, 3.0),
    "hyphenation_min_word_length": (4, 10),
    "badness_exponent": (2.0, 4.0),
    "padding_pixels": (2, 12),
    "supersampling_factor": (1, 16),
    "outside_text_osb_confidence": (0.0, 1.0),
    "outside_text_bbox_expansion_percent_width": (0.0, 1.0),
    "outside_text_bbox_expansion_percent_height": (0.0, 1.0),
    "outside_text_osb_render_expansion_narrow_multiplier": (1.0, 3.0),
    "outside_text_osb_render_expansion_tiny_multiplier": (1.0, 3.0),
    "outside_text_osb_render_expansion_aspect_ratio_threshold": (0.05, 1.0),
    "outside_text_osb_render_expansion_area_ratio_threshold": (0.0, 0.05),
    "outside_text_text_box_proximity_ratio": (0.01, 0.1),
    "outside_text_page_filter_margin_threshold": (0.0, 0.3),
    "outside_text_page_filter_min_area_ratio": (0.0, 0.2),
    "outside_text_min_area_ignore_ratio": (0.0, 0.05),
    "outside_text_osb_max_font_size": (5, 96),
    "outside_text_osb_min_font_size": (5, 96),
    "outside_text_osb_line_spacing": (0.5, 2.0),
    "outside_text_osb_padding_pixels": (2, 12),
    "outside_text_osb_vertical_line_spacing_mult": (0.5, 2.0),
    "outside_text_osb_vertical_font_size_mult": (0.1, 3.0),
    "outside_text_osb_outline_ratio": (0.0, 0.3),
    "jpeg_quality": (1, 100),
    "png_compression": (0, 6),
    "parallel_requests": (1, 20),
}

# Attribute paths for clamping config objects (MangaTranslatorConfig and children)
_CONFIG_ATTR_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "confidence": (("detection", "confidence"),),
    "conjoined_confidence": (("detection", "conjoined_confidence"),),
    "panel_confidence": (("detection", "panel_confidence"),),
    "thresholding_value": (("cleaning", "thresholding_value"),),
    "roi_shrink_px": (("cleaning", "roi_shrink_px"),),
    "temperature": (("translation", "temperature"),),
    "top_p": (("translation", "top_p"),),
    "top_k": (("translation", "top_k"),),
    "max_tokens": (("translation", "max_tokens"),),
    "bubble_min_side_pixels": (("translation", "bubble_min_side_pixels"),),
    "context_image_max_side_pixels": (
        ("translation", "context_image_max_side_pixels"),
    ),
    "previous_context_text_count": (("translation", "previous_context_text_count"),),
    "osb_min_side_pixels": (("translation", "osb_min_side_pixels"),),
    "max_font_size": (("rendering", "max_font_size"),),
    "min_font_size": (("rendering", "min_font_size"),),
    "line_spacing_mult": (("rendering", "line_spacing_mult"),),
    "vertical_line_spacing_mult": (("rendering", "vertical_line_spacing_mult"),),
    "vertical_font_size_mult": (("rendering", "vertical_font_size_mult"),),
    "hyphenation_min_word_length": (("rendering", "hyphenation_min_word_length"),),
    "badness_exponent": (("rendering", "badness_exponent"),),
    "padding_pixels": (("rendering", "padding_pixels"),),
    "supersampling_factor": (("rendering", "supersampling_factor"),),
    "outside_text_osb_confidence": (("outside_text", "osb_confidence"),),
    "outside_text_bbox_expansion_percent_width": (
        ("outside_text", "bbox_expansion_percent_width"),
    ),
    "outside_text_bbox_expansion_percent_height": (
        ("outside_text", "bbox_expansion_percent_height"),
    ),
    "outside_text_osb_render_expansion_narrow_multiplier": (
        ("outside_text", "osb_render_expansion_narrow_multiplier"),
    ),
    "outside_text_osb_render_expansion_tiny_multiplier": (
        ("outside_text", "osb_render_expansion_tiny_multiplier"),
    ),
    "outside_text_osb_render_expansion_aspect_ratio_threshold": (
        ("outside_text", "osb_render_expansion_aspect_ratio_threshold"),
    ),
    "outside_text_osb_render_expansion_area_ratio_threshold": (
        ("outside_text", "osb_render_expansion_area_ratio_threshold"),
    ),
    "outside_text_text_box_proximity_ratio": (
        ("outside_text", "text_box_proximity_ratio"),
    ),
    "outside_text_page_filter_margin_threshold": (
        ("outside_text", "page_filter_margin_threshold"),
    ),
    "outside_text_page_filter_min_area_ratio": (
        ("outside_text", "page_filter_min_area_ratio"),
    ),
    "outside_text_min_area_ignore_ratio": (("outside_text", "min_area_ignore_ratio"),),
    "outside_text_osb_max_font_size": (("outside_text", "osb_max_font_size"),),
    "outside_text_osb_min_font_size": (("outside_text", "osb_min_font_size"),),
    "outside_text_osb_line_spacing": (("outside_text", "osb_line_spacing"),),
    "outside_text_osb_padding_pixels": (("outside_text", "osb_padding_pixels"),),
    "outside_text_osb_vertical_line_spacing_mult": (
        ("outside_text", "osb_vertical_line_spacing_mult"),
    ),
    "outside_text_osb_vertical_font_size_mult": (
        ("outside_text", "osb_vertical_font_size_mult"),
    ),
    "outside_text_osb_outline_ratio": (("outside_text", "osb_outline_ratio"),),
    "jpeg_quality": (("output", "jpeg_quality"),),
    "png_compression": (("output", "png_compression"),),
    "image_upscale_factor": (
        ("preprocessing", "factor"),
    ),
    "parallel_requests": (("parallel_requests",),),
}


def _clamp_numeric(value: Any, min_value: float, max_value: float) -> Any:
    """Clamp numeric values to a min/max range, preserving int type where reasonable."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        clamped = max(min_value, min(value, max_value))
        if isinstance(value, int):
            return int(clamped)
        return clamped
    return value


def clamp_settings(settings: Any) -> Any:
    """
    Clamp numeric settings to UI-defined min/max constraints.

    Supports both flat dictionaries (UI settings/config.json) and MangaTranslatorConfig
    objects (including nested dataclasses). Returns the same object/dict for chaining.
    """

    # Dictionary path: flat keys matching SETTING_CONSTRAINTS
    if isinstance(settings, dict):
        clamped = settings.copy()
        for key, (min_val, max_val) in SETTING_CONSTRAINTS.items():
            if key in clamped:
                clamped[key] = _clamp_numeric(clamped[key], min_val, max_val)
        return clamped

    # Object path: traverse attribute paths if present
    for key, paths in _CONFIG_ATTR_PATHS.items():
        if not hasattr(settings, "__dict__"):
            break
        min_val, max_val = SETTING_CONSTRAINTS[key]
        for path in paths:
            target = settings
            for attr in path[:-1]:
                target = getattr(target, attr, None)
                if target is None:
                    break
            else:
                leaf = path[-1]
                if hasattr(target, leaf):
                    current = getattr(target, leaf)
                    clamped_val = _clamp_numeric(current, min_val, max_val)
                    setattr(target, leaf, clamped_val)
    return settings


def autodetect_yolo_model_path(
    models_dir: Path, bubble_detector_model: str = "yolo_2"
) -> Path:
    """Returns the path for the primary YOLO speech bubble model.

    This function provides a consistent path for the model, which will be
    auto-downloaded by the ModelManager if it doesn't exist. It does not
    validate file existence here. This function previously scanned for any .pt
    file, but now returns a deterministic path to align with auto-downloading.
    """
    yolo_dir = models_dir / "detection" / "yolo"
    if bubble_detector_model == "yolo_2":
        return yolo_dir / "manga109-segmentation-bubble.pt"
    return yolo_dir / "yolov8m_seg-speech-bubble.pt"


