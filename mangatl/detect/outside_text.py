"""Text outside bubbles: detection, page-number and size filters, box expansion, recovery of
missed text, and the per-page work record that cleaning and typesetting use."""

import base64
import re
import unicodedata
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from mangatl.clean.glyph_mask import THRESHOLD as GLYPH_THRESHOLD
from mangatl.clean.glyph_mask import (
    glyph_dilation_radius,
    group_glyphs,
    texts_sharing_detections,
    verified_glyph_box,
)
from mangatl.clean.page_glyph_map import PageGlyphMap
from mangatl.common.exceptions import CancellationError
from mangatl.common.image_utils import (
    cv2_to_pil,
    pil_to_cv2,
    process_bubble_image_cached,
    resize_lanczos_antiring,
    upscaler_for,
)
from mangatl.common.log import log_message
from mangatl.detect.outside_text_detector import OutsideTextDetector
from mangatl.detect.tilt import detect_crop_tilt_angle
from mangatl.ocr import line_ocr
from mangatl.ocr.readers import extract_text_with_manga_ocr
from mangatl.settings import Settings

OSB_EXPANSION_PIXEL_BUFFER = 5  # for bubbles, nearby OSB regions, panels
OSB_SOLID_RATIO_THRESHOLD = 0.95  # Stricter than interior sampling because border ring has no text mask noise
OSB_COLOR_TOLERANCE = 15
OSB_EXPANSION_PX = 2
OSB_WHITE_SNAP_THRESH = 245
OSB_BLACK_SNAP_THRESH = 10
# Share of strongly colored pixels that marks a color page; B/W scans with stray tints
# measured at most 1%, color pages 8% or more
OSB_COLOR_PAGE_MIN_FRACTION = 0.05
# Share of a color page's glyph cores that must be strongly colored for the letters to
# keep that hue; colored SFX measured 0.79-0.99, black text whose glyph map spills onto
# tinted paper, and mixed red and black captions, at most 0.36
OSB_GLYPH_HUE_MIN_SHARE = 0.6
# OSB model confidence on crops that confirms a text region the page-level pass missed;
# on 63 sample pages it kept 31 of 44 missed texts and 2 of 60 non-text regions
OSB_RECOVERY_CONFIDENCE = 0.2
# The page-level OSB confidence (the default) that OSB_RECOVERY_CONFIDENCE was tuned at
OSB_RECOVERY_REFERENCE_CONFIDENCE = 0.5


@dataclass
class OutsideTextWork:
    """Prepared outside-text state: translation data ready, inpaint still pending."""

    pil_image: Image.Image
    config: Settings
    image_path: str | Path
    image_format: str | None
    verbose: bool
    outside_text_results: list
    raw_outside_text_results: list
    original_text_colors: dict
    total_bubble_mask: Any
    outside_detector: Any
    mask_groups: list
    img_w: int
    img_h: int
    mime_type: str
    cv2_ext: str
    outside_text_data: list[dict[str, Any]] = field(default_factory=list)
    is_color_page: bool = False
    glyph_map: PageGlyphMap | None = None

    def __post_init__(self):
        if self.glyph_map is None:
            self.glyph_map = PageGlyphMap.for_page(self.pil_image, self.config, self.verbose)


def outside_text_crop(
    original_cv_image: np.ndarray,
    bbox: tuple[int, int, int, int],
    raw_bbox: tuple[int, ...],
    config: Settings,
    verbose: bool,
) -> np.ndarray:
    """The image of one outside-text region read by OCR and sent to the LLM: the detection (raw
    box, falling back to the expanded box) cut from the page before inpainting, upscaled per
    config.ocr.upscale."""
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = raw_bbox
    rx1 = max(0, min(original_cv_image.shape[1], rx1))
    ry1 = max(0, min(original_cv_image.shape[0], ry1))
    rx2 = max(0, min(original_cv_image.shape[1], rx2))
    ry2 = max(0, min(original_cv_image.shape[0], ry2))
    if rx2 > rx1 and ry2 > ry1:
        outside_text_image_cv = original_cv_image[ry1:ry2, rx1:rx2].copy()
    else:
        outside_text_image_cv = original_cv_image[y1:y2, x1:x2].copy()
    outside_text_image_pil = cv2_to_pil(outside_text_image_cv)
    osb_upscale_method = config.ocr.upscale

    if osb_upscale_method in ("model", "model_lite"):
        final_text_pil = process_bubble_image_cached(
            outside_text_image_pil,
            upscaler_for(config.models, osb_upscale_method),
            config.ocr.outside_min_side,
            "min",
            osb_upscale_method,
            verbose,
        )
    elif osb_upscale_method == "lanczos":
        w, h = outside_text_image_pil.size
        min_side = min(w, h)
        if min_side < config.ocr.outside_min_side:
            scale_factor = config.ocr.outside_min_side / min_side
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            resized_text = resize_lanczos_antiring(outside_text_image_pil, (new_w, new_h))
        else:
            resized_text = outside_text_image_pil
        final_text_pil = resized_text
    else:
        final_text_pil = outside_text_image_pil

    return pil_to_cv2(final_text_pil)


def _build_outside_text_data(
    pil_image: Image.Image,
    outside_text_results,
    raw_outside_text_results,
    original_text_colors,
    extracted_text_colors,
    none_skipped_clip_bboxes,
    mime_type: str,
    cv2_ext: str,
    config: Settings,
    verbose: bool,
    glyph_map: PageGlyphMap | None,
) -> list[dict[str, Any]]:
    outside_text_data = []
    original_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    for expanded_res, raw_res in zip(outside_text_results, raw_outside_text_results, strict=False):
        bbox_coords, conf = expanded_res
        raw_coords, _ = raw_res

        x1, y1, x2, y2 = [int(c) for c in bbox_coords]
        bbox_tuple = (x1, y1, x2, y2)
        raw_bbox_tuple = tuple(int(c) for c in raw_coords)
        rx1, ry1, rx2, ry2 = raw_bbox_tuple

        rx1 = max(0, min(original_cv_image.shape[1], rx1))
        ry1 = max(0, min(original_cv_image.shape[0], ry1))
        rx2 = max(0, min(original_cv_image.shape[1], rx2))
        ry2 = max(0, min(original_cv_image.shape[0], ry2))
        original_crop_pil = cv2_to_pil(original_cv_image[y1:y2, x1:x2].copy())
        outside_text_image_cv = outside_text_crop(
            original_cv_image, bbox_tuple, raw_bbox_tuple, config, verbose
        )

        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect_ratio = float(h) / float(w)

        needs_text_bg = False
        if none_skipped_clip_bboxes:
            bcx = (x1 + x2) / 2
            bcy = (y1 + y2) / 2
            for clip_bbox in none_skipped_clip_bboxes:
                cx1, cy1, cx2, cy2 = clip_bbox
                if cx1 <= bcx <= cx2 and cy1 <= bcy <= cy2:
                    needs_text_bg = True
                    break

        angle_deg, tilt_conf, obb_dims, tilt_axis = (
            0.0,
            0.0,
            (float(w), float(h)),
            "none",
        )
        osb_tilt_estimation_side_pixels = 256
        if config.typesetting.outside.follow_tilt and rx2 > rx1 and ry2 > ry1:
            glyph_prob = None if glyph_map is None else glyph_map.get()
            angle_deg, tilt_conf, obb_dims, tilt_axis = detect_crop_tilt_angle(
                crop_cv=original_cv_image[ry1:ry2, rx1:rx2],
                # original_text_colors flags a dark background, i.e. light ink
                is_dark_text=not original_text_colors.get(raw_bbox_tuple, False),
                max_side=osb_tilt_estimation_side_pixels,
                deadband_deg=config.typesetting.outside.min_tilt,
                max_tilt_deg=config.typesetting.outside.max_tilt,
                ink_mask=(None if glyph_prob is None else glyph_prob[ry1:ry2, rx1:rx2] > GLYPH_THRESHOLD),
            )

        try:
            is_success, buffer = cv2.imencode(cv2_ext, outside_text_image_cv)
            if is_success:
                image_b64 = base64.b64encode(buffer).decode("utf-8")
                outside_text_data.append(
                    {
                        "bbox": bbox_tuple,
                        "original_bbox": raw_bbox_tuple,
                        "confidence": conf,
                        "is_outside_text": True,
                        "image_b64": image_b64,
                        "mime_type": mime_type,
                        "is_dark_text": original_text_colors.get(raw_bbox_tuple, True),
                        "text_color_rgb": extracted_text_colors.get(raw_bbox_tuple),
                        "aspect_ratio": aspect_ratio,
                        "needs_text_background": needs_text_bg,
                        "original_crop_pil": original_crop_pil,
                        "tilt_deg": angle_deg,
                        "tilt_conf": tilt_conf,
                        "tilt_axis": tilt_axis,
                        "oriented_dimensions": obb_dims,
                    }
                )
        except Exception as e:
            log_message(
                f"Error encoding outside text bbox {(x1, y1, x2, y2)}: {e}",
                verbose=verbose,
            )

    if outside_text_data:  # line-by-line OCR finds this page's text lines once (line_ocr)
        page_lines = line_ocr.PageLines(pil_image)
        for item in outside_text_data:
            item["mt_page"] = page_lines
    return outside_text_data


def apply_inpaint_render_metadata(
    outside_text_data: list[dict[str, Any]],
    extracted_text_colors: dict,
    none_skipped_clip_bboxes: set,
) -> None:
    """Update existing OSB entries in-place with render-only fields from inpainting."""
    for item in outside_text_data:
        raw_bbox_tuple = item.get("original_bbox")
        bbox = item.get("bbox")
        if raw_bbox_tuple is not None and raw_bbox_tuple in extracted_text_colors:
            item["text_color_rgb"] = extracted_text_colors[raw_bbox_tuple]
        elif item.get("text_color_rgb") is None and extracted_text_colors and bbox:
            x1, y1, x2, y2 = bbox
            bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
            for key, color in extracted_text_colors.items():
                if key in (raw_bbox_tuple, bbox):
                    item["text_color_rgb"] = color
                    break
                if (
                    isinstance(key, tuple)
                    and len(key) == 4
                    and key[0] <= bcx <= key[2]
                    and key[1] <= bcy <= key[3]
                ):
                    item["text_color_rgb"] = color
                    break

        needs_text_bg = False
        if none_skipped_clip_bboxes and bbox is not None:
            x1, y1, x2, y2 = bbox
            bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
            for clip_bbox in none_skipped_clip_bboxes:
                cx1, cy1, cx2, cy2 = clip_bbox
                if cx1 <= bcx <= cx2 and cy1 <= bcy <= cy2:
                    needs_text_bg = True
                    break
        item["needs_text_background"] = needs_text_bg


def paint_mask(image: Image.Image, mask: np.ndarray, color) -> Image.Image:
    """Copy of `image` with `color` painted wherever `mask` is set."""
    filled = image.copy()
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return filled
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    region = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), mode="L")
    filled.paste(Image.new("RGB", (x1 - x0, y1 - y0), color), (x0, y0), mask=region)
    return filled


def results_bounds(results: list, indices: list[int]) -> tuple[int, int, int, int]:
    """Union (x0, y0, x1, y1) of the boxes of `results[i]` for each index."""
    boxes = [results[i][0] for i in indices]
    return (
        int(min(box[0] for box in boxes)),
        int(min(box[1] for box in boxes)),
        int(max(box[2] for box in boxes)),
        int(max(box[3] for box in boxes)),
    )


FLAT_P90_MAX = 6  # paper measured 2-6; blurred dark art 9-10


def border_color(pixels: np.ndarray) -> tuple[tuple[int, int, int], float]:
    """Median RGB of border pixels (snapped to pure white/black) and the share near it.

    A border only counts as solid when it is truly flat (90% of pixels within FLAT_P90_MAX of
    the median): blurred dark art passes the 95%-within-15-levels test and got hard black boxes,
    so anything less flat is reported just under the solid threshold (sent to inpainting).
    """
    median = np.median(pixels, axis=0).astype(int)
    diffs = np.max(np.abs(pixels.astype(int) - median), axis=1)
    solid_ratio = float(np.mean(diffs <= OSB_COLOR_TOLERANCE))
    if np.all(median >= OSB_WHITE_SNAP_THRESH):
        color = (255, 255, 255)
    elif np.all(median <= OSB_BLACK_SNAP_THRESH):
        color = (0, 0, 0)
    else:
        color = (int(median[0]), int(median[1]), int(median[2]))
    if float(np.percentile(diffs, 90)) > FLAT_P90_MAX:
        solid_ratio = min(solid_ratio, OSB_SOLID_RATIO_THRESHOLD - 0.01)
    return color, solid_ratio


class Glyph(NamedTuple):
    """A solid component whose border mostly contrasts."""

    area: int
    ratio: float  # share of the border that contrasts
    on_frame: bool  # touches the crop edge, so it may be background
    x: int
    y: int
    mask: np.ndarray  # component pixels within its bounding box at (x, y)


def outlined_glyph_components(
    crop_gray: np.ndarray,
    glyph_mask: np.ndarray,
    is_contrasting,
    min_core_ratio: float = 0.0,
) -> list[Glyph]:
    """Each solid glyph in `glyph_mask` whose border mostly contrasts."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(glyph_mask)
    height, width = crop_gray.shape
    found = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 10:
            continue
        # Work inside the component's own box, padded for the 5x5 dilation; scanning
        # the whole region per component takes seconds on screentone
        x, y, w, h = (int(v) for v in stats[label, :4])
        y0, y1 = max(0, y - 3), min(height, y + h + 3)
        x0, x1 = max(0, x - 3), min(width, x + w + 3)
        comp = (labels[y0:y1, x0:x1] == label).astype(np.uint8) * 255
        core = np.count_nonzero(cv2.erode(comp, np.ones((2, 2), np.uint8)))
        if core == 0 or core / area < min_core_ratio:
            continue
        dilated = cv2.dilate(comp, np.ones((5, 5), np.uint8))
        boundary = cv2.bitwise_and(dilated, cv2.bitwise_not(comp))
        border = crop_gray[y0:y1, x0:x1][boundary == 255]
        if len(border) == 0:
            continue
        ratio = np.count_nonzero(is_contrasting(border)) / len(border)
        if ratio >= 0.40:
            on_frame = x == 0 or y == 0 or x + w >= width or y + h >= height
            mask = labels[y : y + h, x : x + w] == label
            found.append(Glyph(area, ratio, on_frame, x, y, mask))
    return found


def _is_page_number(text: str) -> bool:
    """Match digits, "Page 20", "p. 20", "- 20 -", etc."""
    # manga-ocr returns full-width characters; NFKC folds them to ASCII
    text = unicodedata.normalize("NFKC", text)
    return bool(
        re.match(
            r"^\s*[-–—―ー]?\s*(?:page\.?|p\.?)?\s*\d+\s*[-–—―ー]?\s*$",
            text,
            re.IGNORECASE,
        )
    )


def _drop_small_osb_regions(
    outside_text_results: list, min_ignore: float, img_w: int, img_h: int, verbose: bool
) -> list:
    """Drop OSB detections smaller than `min_ignore` of the image area."""
    image_area = float(img_w * img_h)
    kept_results = []
    removed_count = 0
    for bbox, conf in outside_text_results:
        x1, y1, x2, y2 = bbox
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, image_area)
        if area_ratio < min_ignore:
            removed_count += 1
            log_message(
                f"Skipping small OSB region (area {area_ratio:.4%} < {min_ignore:.4%})",
                verbose=verbose,
            )
            continue
        kept_results.append((bbox, conf))
    outside_text_results = kept_results
    if removed_count:
        log_message(
            f"Filtered {removed_count} OSB region(s) below min area threshold "
            f"({min_ignore:.2%} of image); {len(outside_text_results)} remaining",
            verbose=verbose,
        )
    return outside_text_results


def _drop_page_numbers(
    pil_image: Image.Image,
    outside_text_results: list,
    config: Settings,
    img_w: int,
    img_h: int,
    verbose: bool,
) -> list:
    """Drop OSB detections that OCR as page numbers.

    Only small detections in the top/bottom margin are OCR'd."""
    suspicious_crops = []
    suspicious_indices = []
    safe_results = []

    margin_threshold = max(0.0, min(0.3, config.detection.outside_text.page_filter_margin))
    min_area_threshold = max(0.0, min(0.2, config.detection.outside_text.page_filter_min_area))

    for i, res in enumerate(outside_text_results):
        bbox, _ = res
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cy = (y1 + y2) / 2

        is_in_margin = (cy < img_h * margin_threshold) or (cy > img_h * (1 - margin_threshold))

        area = (x2 - x1) * (y2 - y1)
        is_small = area < (img_w * img_h * min_area_threshold)

        if is_in_margin and is_small:
            suspicious_crops.append(pil_image.crop((x1, y1, x2, y2)))
            suspicious_indices.append(i)
        else:
            safe_results.append(res)

    if suspicious_crops:
        log_message(
            f"Verifying {len(suspicious_crops)} suspicious OSB regions with OCR...",
            verbose=verbose,
        )
        suspicious_texts = extract_text_with_manga_ocr(suspicious_crops, config.models, verbose=verbose)

        for i, text in enumerate(suspicious_texts):
            if not _is_page_number(text):
                safe_results.append(outside_text_results[suspicious_indices[i]])
            else:
                log_message(f"Filtered out page number: '{text}'", verbose=verbose)

        outside_text_results = safe_results
        log_message(
            f"Remaining OSB regions after filtering: {len(outside_text_results)}",
            verbose=verbose,
        )
    return outside_text_results


def _expand_osb_bboxes(
    outside_text_results: list,
    config: Settings,
    img_w: int,
    img_h: int,
    bubble_data: list[dict[str, Any]] | None,
    panels: list[tuple[int, int, int, int]] | None,
    verbose: bool,
) -> list:
    """Grow narrow/tall and tiny OSB boxes (which render too small) within their
    panel, without overlapping bubbles or other OSB boxes."""
    style = config.typesetting.outside
    narrow_expansion_mult = style.expansion_narrow
    tiny_expansion_mult = style.expansion_tiny
    aspect_ratio_threshold = style.expansion_aspect_threshold
    area_ratio_threshold = style.expansion_area_threshold
    if max(narrow_expansion_mult, tiny_expansion_mult) <= 1.0:
        return outside_text_results
    log_message(
        f"Expanding OSB bboxes (narrow/tall: {narrow_expansion_mult}x, tiny: {tiny_expansion_mult}x)...",
        verbose=verbose,
    )
    expanded_results = []
    for i, res in enumerate(outside_text_results):
        bbox, conf = res
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        aspect_ratio = float(w) / float(max(1, h))
        area_ratio = (w * h) / float(max(1, img_w * img_h))
        is_narrow_tall = aspect_ratio <= aspect_ratio_threshold
        is_tiny = area_ratio < area_ratio_threshold

        expansion_mult = 1.0
        if is_narrow_tall:
            expansion_mult = max(expansion_mult, narrow_expansion_mult)
        if is_tiny:
            expansion_mult = max(expansion_mult, tiny_expansion_mult)

        if expansion_mult <= 1.0:
            expanded_results.append(([int(x1), int(y1), int(x2), int(y2)], conf))
            continue

        cx = x1 + w / 2
        cy = y1 + h / 2

        new_w = w * expansion_mult
        new_h = h * expansion_mult

        nx1 = int(cx - new_w / 2)
        ny1 = int(cy - new_h / 2)
        nx2 = int(cx + new_w / 2)
        ny2 = int(cy + new_h / 2)

        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(img_w, nx2)
        ny2 = min(img_h, ny2)

        if panels:
            associated_panel = None
            for p in panels:
                px1, py1, px2, py2 = p
                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    associated_panel = p
                    break
            if associated_panel:
                px1, py1, px2, py2 = associated_panel
                panel_buffer = OSB_EXPANSION_PIXEL_BUFFER
                c_px1 = min(int(px1) + panel_buffer, int(px2))
                c_py1 = min(int(py1) + panel_buffer, int(py2))
                c_px2 = max(int(px2) - panel_buffer, int(px1))
                c_py2 = max(int(py2) - panel_buffer, int(py1))

                nx1 = max(c_px1, nx1)
                ny1 = max(c_py1, ny1)
                nx2 = min(c_px2, nx2)
                ny2 = min(c_py2, ny2)

        buffer = OSB_EXPANSION_PIXEL_BUFFER
        obstacles = []
        if bubble_data:
            for b in bubble_data:
                bb = b.get("bbox")
                if bb and len(bb) == 4:
                    bx1, by1, bx2, by2 = [int(c) for c in bb]
                    obstacles.append(
                        (
                            max(0, bx1 - buffer),
                            max(0, by1 - buffer),
                            min(img_w, bx2 + buffer),
                            min(img_h, by2 + buffer),
                        )
                    )

        for j, other_res in enumerate(outside_text_results):
            if i == j:
                continue

            if j < i:
                ob, _ = expanded_results[j]
            else:
                ob, _ = other_res
            obx1, oby1, obx2, oby2 = [int(c) for c in ob]

            obstacles.append(
                (
                    max(0, obx1 - buffer),
                    max(0, oby1 - buffer),
                    min(img_w, obx2 + buffer),
                    min(img_h, oby2 + buffer),
                )
            )

        for ox1, oy1, ox2, oy2 in obstacles:
            if not (nx2 <= ox1 or nx1 >= ox2 or ny2 <= oy1 or ny1 >= oy2):
                can_retract_nx2 = (nx2 - ox1) if (ox1 >= x2) else float("inf")
                can_retract_nx1 = (ox2 - nx1) if (ox2 <= x1) else float("inf")
                can_retract_ny2 = (ny2 - oy1) if (oy1 >= y2) else float("inf")
                can_retract_ny1 = (oy2 - ny1) if (oy2 <= y1) else float("inf")

                min_retract = min(
                    can_retract_nx2,
                    can_retract_nx1,
                    can_retract_ny2,
                    can_retract_ny1,
                )
                if min_retract != float("inf"):
                    if min_retract == can_retract_nx2:
                        nx2 = ox1
                    elif min_retract == can_retract_nx1:
                        nx1 = ox2
                    elif min_retract == can_retract_ny2:
                        ny2 = oy1
                    elif min_retract == can_retract_ny1:
                        ny1 = oy2

        nx1 = min(nx1, int(x1))
        ny1 = min(ny1, int(y1))
        nx2 = max(nx2, int(x2))
        ny2 = max(ny2, int(y2))

        expanded_results.append(([nx1, ny1, nx2, ny2], conf))

    outside_text_results = expanded_results
    return outside_text_results


def _build_total_bubble_mask(
    bubble_data: list[dict[str, Any]] | None, img_w: int, img_h: int, verbose: bool
) -> np.ndarray:
    """Mask of all detected speech bubbles (dilated), so OSB inpainting avoids them."""
    total_bubble_mask = np.zeros((img_h, img_w), dtype=bool)
    if bubble_data:
        for bubble in bubble_data:
            try:
                mask = bubble.get("seg_mask") if isinstance(bubble, dict) else None
                if mask is not None:
                    mask_np = np.asarray(mask)
                    if mask_np.ndim == 3:
                        mask_np = mask_np[..., 0]
                    mask_bool = mask_np > 0
                    if mask_bool.shape[0] == img_h and mask_bool.shape[1] == img_w:
                        total_bubble_mask |= mask_bool
                        continue

                bbox = bubble.get("bbox") if isinstance(bubble, dict) else None
                if bbox and len(bbox) == 4:
                    x0, y0, x1, y1 = [int(c) for c in bbox]
                    x0 = max(0, min(img_w, x0))
                    x1 = max(0, min(img_w, x1))
                    y0 = max(0, min(img_h, y0))
                    y1 = max(0, min(img_h, y1))
                    if x1 > x0 and y1 > y0:
                        total_bubble_mask[y0:y1, x0:x1] = True
            except Exception as e:
                log_message(
                    f"Warning: Failed to apply bubble mask for OSB exclusion: {e}",
                    verbose=verbose,
                )

        if np.any(total_bubble_mask):
            # Dilate the bubble mask to provide a safe buffer for OSB fill
            kernel = np.ones((11, 11), np.uint8)
            total_bubble_mask = cv2.dilate(total_bubble_mask.astype(np.uint8), kernel, iterations=1).astype(
                bool
            )
    return total_bubble_mask


def _probe_original_text_colors(
    pil_image: Image.Image, raw_outside_text_results: list, verbose: bool
) -> dict:
    """Map each OSB box to whether its background is dark (via 2-color K-Means)."""
    original_text_colors = {}
    for ocr_result in raw_outside_text_results:
        bbox_coords, _ = ocr_result
        x1, y1, x2, y2 = [int(c) for c in bbox_coords]
        bbox_tuple = (x1, y1, x2, y2)

        bbox_area_img = pil_image.crop((x1, y1, x2, y2))
        bbox_array = np.array(bbox_area_img)

        if bbox_array.shape[-1] == 4:
            bbox_array = bbox_array[..., :3]

        pixels = bbox_array.reshape(-1, 3)
        # Every 4th pixel is plenty to find two colors, at a tenth of the cost
        if len(pixels) >= 8:
            pixels = pixels[::4]

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=1)
        kmeans.fit(pixels)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        unique, counts = np.unique(labels, return_counts=True)
        dominant_cluster_idx = unique[np.argmax(counts)]

        # Dominant cluster is usually the background (text pixels are sparse)
        bg_color_rgb = centers[dominant_cluster_idx]
        # Use proper luminance calculation (ITU-R BT.601)
        bg_brightness = 0.299 * bg_color_rgb[0] + 0.587 * bg_color_rgb[1] + 0.114 * bg_color_rgb[2]
        is_dark_text = bg_brightness < 128  # passed downstream; renderer inverts for text color
        original_text_colors[bbox_tuple] = is_dark_text

        log_message(
            f"OSB bbox {bbox_tuple}: "
            f"{'Dark' if is_dark_text else 'Light'} background detected "
            f"(luminance={bg_brightness:.1f})",
            verbose=verbose,
        )
    return original_text_colors


def _recovers_missed_text(config: Settings) -> bool:
    # Caption-box-only mode deliberately ignores the SFX that recovery mostly finds
    return bool(
        config.detection.outside_text.recover_missed_text
        and config.cleaning.text_segmentation
        and not config.detection.outside_text.text_free_only
    )


def _recovery_confidence(osb_confidence: float) -> float:
    """Crop confidence that confirms recovered text for a page-level `osb_confidence`.

    Above the reference it keeps the same share of the distance to 1.0, so raising OSB
    confidence also tightens recovery; below it, lower crop scores would mostly add art.
    """
    raised = max(0.0, osb_confidence - OSB_RECOVERY_REFERENCE_CONFIDENCE)
    headroom = (1.0 - OSB_RECOVERY_CONFIDENCE) / (1.0 - OSB_RECOVERY_REFERENCE_CONFIDENCE)
    return min(1.0, OSB_RECOVERY_CONFIDENCE + raised * headroom)


def _recover_missed_osb_text(
    pil_image: Image.Image,
    outside_text_results: list,
    bubble_data: list[dict[str, Any]] | None,
    glyph_prob: np.ndarray,
    detector: OutsideTextDetector,
    verbose: bool,
    confidence: float = OSB_RECOVERY_CONFIDENCE,
    rejected_results: list | None = None,
) -> list:
    """OSB results for text the OSB model missed: glyph-map letters outside every
    bubble and OSB box, kept where the OSB model finds text on crops around them at
    `confidence`, and joined where one of its boxes spans several of them, like the
    words of a caption.

    `rejected_results` are the page-level detections at the default confidence, so
    text a raised OSB confidence filtered out isn't recovered: the model saw it."""
    img_w, img_h = pil_image.size
    radius = glyph_dilation_radius((img_w, img_h))
    covered = np.zeros((img_h, img_w), bool)
    for bubble in bubble_data or []:
        mask = bubble.get("seg_mask")
        if mask is not None:
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask[..., 0]
            if mask.shape == covered.shape:
                covered |= mask > 0
    covered = cv2.dilate(covered.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8)).astype(bool)
    for bubble in bubble_data or []:
        bbox = bubble.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (int(c) for c in bbox)
        covered[max(0, y0) : y1, max(0, x0) : x1] = True
    # Glyphs of found text spill past its box, so pad the box by the glyph dilation
    pad = 2 * radius
    for bbox, _ in [*outside_text_results, *(rejected_results or [])]:
        x0, y0, x1, y1 = (int(c) for c in bbox)
        covered[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad] = True

    groups = group_glyphs((glyph_prob > GLYPH_THRESHOLD) & ~covered)
    if not groups:
        return []
    image_cv = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    words = []
    near = detector.detect_text_near(image_cv, [group.box for group in groups], confidence)
    for group, detections in zip(groups, near, strict=True):
        box = verified_glyph_box(group, [d for d, _ in detections])
        if box is not None:
            words.append((box, max(conf for _, conf in detections)))
    recovered = []
    all_detections = [d for detections in near for d, _ in detections]
    for text in texts_sharing_detections([box for box, _ in words], all_detections):
        boxes = np.array([words[i][0] for i in text], dtype=np.float32)
        recovered.append(
            (
                np.concatenate([boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]),
                max(words[i][1] for i in text),
            )
        )
    log_message(
        f"Recovered {len(recovered)} OSB text region(s) the OSB model missed ({len(groups)} checked)",
        always_print=bool(recovered),
        verbose=verbose,
    )
    return recovered


def prepare_outside_text_work(
    pil_image: Image.Image,
    config: Settings,
    image_path: str | Path,
    image_format: str | None,
    verbose: bool = False,
    bubble_data: list[dict[str, Any]] | None = None,
    text_free_boxes: list[list[float]] | None = None,
    panels: list[tuple[int, int, int, int]] | None = None,
    glyph_map: PageGlyphMap | None = None,
) -> OutsideTextWork | None:
    """Detect outside text and build translation crops without inpainting.

    `glyph_map` is the original page's text map, shared with bubble cleaning; one is
    made for `pil_image` without it.
    """
    if not config.detection.outside_text.enabled:
        return None

    log_message("Detecting text outside speech bubbles...", verbose=verbose)

    try:
        outside_detector = OutsideTextDetector(config.models)
        detect = partial(
            outside_detector.detect_outside_text,
            str(image_path),
            conjoined_confidence=config.detection.conjoined_confidence,
            verbose=verbose,
            image_override=pil_image,
            existing_bubbles=bubble_data,
            text_free_boxes=text_free_boxes,
            bubble_detector_model=config.detection.bubble_model,
            min_area_ignore_ratio=config.detection.outside_text.min_area_ignore_ratio,
            text_free_only=config.detection.outside_text.text_free_only,
        )
        osb_confidence = config.detection.outside_text.confidence
        outside_text_results = detect(confidence=osb_confidence)

        if glyph_map is None:
            glyph_map = PageGlyphMap.for_page(pil_image, config, verbose)
        if _recovers_missed_text(config):
            glyph_prob = glyph_map.get()
            if glyph_prob is not None:
                try:
                    # Text a raised OSB confidence filtered out stays out
                    rejected = (
                        detect(confidence=OSB_RECOVERY_REFERENCE_CONFIDENCE)
                        if osb_confidence > OSB_RECOVERY_REFERENCE_CONFIDENCE
                        else []
                    )
                    outside_text_results = list(outside_text_results) + _recover_missed_osb_text(
                        pil_image,
                        outside_text_results,
                        bubble_data,
                        glyph_prob,
                        outside_detector,
                        verbose,
                        confidence=_recovery_confidence(osb_confidence),
                        rejected_results=rejected,
                    )
                except CancellationError:
                    raise
                except Exception as e:
                    # The OSB model's own results are still usable
                    log_message(f"Missed OSB text recovery failed: {e}", always_print=True)

        if not outside_text_results:
            log_message("No outside text regions found", verbose=verbose)
            return None

        img_w, img_h = pil_image.size

        min_ignore = max(0.0, min(0.05, config.detection.outside_text.min_area_ignore_ratio))
        if min_ignore > 0.0:
            outside_text_results = _drop_small_osb_regions(
                outside_text_results, min_ignore, img_w, img_h, verbose
            )
            if not outside_text_results:
                log_message(
                    "No outside text regions remaining after min area filter",
                    verbose=verbose,
                )
                return None

        if config.detection.outside_text.filter_page_numbers and outside_text_results:
            outside_text_results = _drop_page_numbers(
                pil_image, outside_text_results, config, img_w, img_h, verbose
            )

        raw_outside_text_results = outside_text_results.copy()

        outside_text_results = _expand_osb_bboxes(
            outside_text_results, config, img_w, img_h, bubble_data, panels, verbose
        )
        total_bubble_mask = _build_total_bubble_mask(bubble_data, img_w, img_h, verbose)

        mime_type = "image/png" if image_format and image_format.upper() == "PNG" else "image/jpeg"
        cv2_ext = ".png" if image_format and image_format.upper() == "PNG" else ".jpg"

        original_text_colors = _probe_original_text_colors(pil_image, raw_outside_text_results, verbose)

        mask_groups, _ = outside_detector.get_text_masks(
            str(image_path),
            bbox_expansion_percent_width=config.detection.outside_text.bbox_expansion_width,
            bbox_expansion_percent_height=config.detection.outside_text.bbox_expansion_height,
            text_box_proximity_ratio=config.detection.outside_text.text_box_proximity_ratio,
            verbose=verbose,
            image_override=pil_image,
            existing_results=raw_outside_text_results,
        )

        outside_text_data = _build_outside_text_data(
            pil_image=pil_image,
            outside_text_results=outside_text_results,
            raw_outside_text_results=raw_outside_text_results,
            original_text_colors=original_text_colors,
            extracted_text_colors={},
            none_skipped_clip_bboxes=set(),
            mime_type=mime_type,
            cv2_ext=cv2_ext,
            config=config,
            verbose=verbose,
            glyph_map=glyph_map,
        )

        return OutsideTextWork(
            pil_image=pil_image,
            config=config,
            image_path=image_path,
            image_format=image_format,
            verbose=verbose,
            outside_text_results=outside_text_results,
            raw_outside_text_results=raw_outside_text_results,
            original_text_colors=original_text_colors,
            total_bubble_mask=total_bubble_mask,
            outside_detector=outside_detector,
            mask_groups=mask_groups,
            img_w=img_w,
            img_h=img_h,
            mime_type=mime_type,
            cv2_ext=cv2_ext,
            outside_text_data=outside_text_data,
            is_color_page=_is_color_page(pil_image),
            glyph_map=glyph_map,
        )

    except Exception as e:
        log_message(
            f"Error during outside text detection: {e}",
            always_print=True,
        )
        return None


def _is_color_page(image: Image.Image) -> bool:
    """Whether enough of the page is strongly colored to use the color-page text detector."""
    hsv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2HSV)
    colorful = (hsv[..., 1] > 60) & (hsv[..., 2] > 40)
    return bool(colorful.mean() >= OSB_COLOR_PAGE_MIN_FRACTION)
