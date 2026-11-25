"""Removing text outside bubbles: flat fills for plain backgrounds, LaMa elsewhere, and the text colour for the translation."""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from PIL import Image

from mangatl.clean.glyph_mask import THRESHOLD as GLYPH_THRESHOLD
from mangatl.clean.inpainting import make_inpainter
from mangatl.clean.page_glyph_map import PageGlyphMap
from mangatl.common.cancellation import raise_if_cancelled
from mangatl.common.coordinator import expanded_mask_bbox, partition_non_overlapping_waves, paste_image_region
from mangatl.common.exceptions import CancellationError
from mangatl.common.log import log_message
from mangatl.detect.outside_text import (
    OSB_EXPANSION_PX,
    OSB_GLYPH_HUE_MIN_SHARE,
    OSB_SOLID_RATIO_THRESHOLD,
    Glyph,
    OutsideTextWork,
    apply_inpaint_render_metadata,
    border_color,
    outlined_glyph_components,
    paint_mask,
    prepare_outside_text_work,
    results_bounds,
)
from mangatl.detect.tilt import compute_effective_tilt_angle
from mangatl.settings import Settings


def _effective_osb_tilt(item: dict[str, Any], config: Settings) -> float:
    return compute_effective_tilt_angle(
        tilt_deg=item.get("tilt_deg", 0.0),
        tilt_conf=item.get("tilt_conf", 0.0),
        follow_tilt=config.typesetting.outside.follow_tilt,
        confidence_threshold=config.typesetting.outside.tilt_confidence,
        min_tilt_deg=config.typesetting.outside.min_tilt,
        max_tilt_deg=config.typesetting.outside.max_tilt,
    )


def _osb_fill_box_mask(img_w: int, img_h: int, box: tuple[int, int, int, int], rot_deg: float) -> np.ndarray:
    """Boolean mask of `box`, or of a shrunken copy turned by `rot_deg` when tilted.

    The turned copy is clipped to `box`: its corners can reach past it, onto art the
    solid-background check never sampled, such as a panel border."""
    bx0, by0, bx1, by1 = box
    mask = np.zeros((img_h, img_w), dtype=bool)
    if abs(rot_deg) <= 0.01:
        mask[by0:by1, bx0:bx1] = True
        return mask
    bcx = (bx0 + bx1) / 2.0
    bcy = (by0 + by1) / 2.0
    shrink = max(0.70, math.cos(math.radians(rot_deg)))
    hw = (bx1 - bx0) * shrink / 2.0
    hh = (by1 - by0) * shrink / 2.0
    rad = math.radians(rot_deg)
    cos_t = math.cos(rad)
    sin_t = math.sin(rad)
    corners = np.array(
        [
            [bcx - hw * cos_t + hh * sin_t, bcy - hw * sin_t - hh * cos_t],
            [bcx + hw * cos_t + hh * sin_t, bcy + hw * sin_t - hh * cos_t],
            [bcx + hw * cos_t - hh * sin_t, bcy + hw * sin_t + hh * cos_t],
            [bcx - hw * cos_t - hh * sin_t, bcy - hw * sin_t + hh * cos_t],
        ],
        dtype=np.int32,
    )
    mask_uint8 = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask_uint8, [corners], 255)
    mask[by0:by1, bx0:bx1] = mask_uint8[by0:by1, bx0:bx1] > 0
    return mask


@dataclass(frozen=True)
class _PendingOsbFill:
    """A solid fill for one OSB text box, painted once its render angle is known."""

    color: tuple[int, int, int]
    box: tuple[int, int, int, int]
    tilt_deg: float  # the detected tilt, for text that is never rendered
    bubble_mask: np.ndarray | None


def paint_pending_osb_fill(
    image: Image.Image, item: dict[str, Any], rotation_deg: float | None = None
) -> Image.Image:
    """Paint and clear `item`'s pending solid fill, turned by `rotation_deg` (default:
    the detected tilt). Returns `image` itself when nothing is pending."""
    fill = item.pop("pending_fill", None)
    if fill is None:
        return image
    rot = fill.tilt_deg if rotation_deg is None else rotation_deg
    mask = _osb_fill_box_mask(image.width, image.height, fill.box, rot)
    if fill.bubble_mask is not None:
        mask &= ~fill.bubble_mask
    return paint_mask(image, mask, fill.color)


def paint_pending_osb_fills(image: Image.Image, items: list[dict[str, Any]]) -> Image.Image:
    """Paint every pending solid fill in `items` at its detected tilt."""
    for item in items:
        image = paint_pending_osb_fill(image, item)
    return image


def _osb_fill_boxes(
    mask_indices: list[int], outside_text_results: list, img_w: int, img_h: int
) -> list[tuple[int, tuple[int, int, int, int]]]:
    """(index, box clipped to the page) for each of the group's non-empty text boxes."""
    boxes = []
    for idx in mask_indices:
        if idx >= len(outside_text_results):
            continue
        coords = outside_text_results[idx][0]
        if isinstance(coords, (list, tuple)) and len(coords) == 1 and isinstance(coords[0], (list, tuple)):
            coords = coords[0]
        bx0 = max(0, int(coords[0]))
        by0 = max(0, int(coords[1]))
        bx1 = min(img_w, int(coords[2]))
        by1 = min(img_h, int(coords[3]))
        if bx1 > bx0 and by1 > by0:
            boxes.append((idx, (bx0, by0, bx1, by1)))
    return boxes


def _build_osb_simple_fill_mask(
    img_w: int,
    img_h: int,
    mask_indices: list[int],
    outside_text_results: list,
    outside_text_data: list[dict[str, Any]],
    config: Settings,
    fallback_bounds: tuple[int | None, int | None, int | None, int | None] | None = None,
    total_bubble_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build a 2D boolean mask for solid-fill inpainting, respecting detected OSB text tilt."""
    rect_mask = np.zeros((img_h, img_w), dtype=bool)
    has_drawn = False

    if mask_indices and outside_text_results:
        for idx, box in _osb_fill_boxes(mask_indices, outside_text_results, img_w, img_h):
            rot_deg = 0.0
            if idx < len(outside_text_data):
                rot_deg = _effective_osb_tilt(outside_text_data[idx], config)
            rect_mask |= _osb_fill_box_mask(img_w, img_h, box, rot_deg)
            has_drawn = True

    elif fallback_bounds:
        b_ox0, b_oy0, b_ox1, b_oy1 = fallback_bounds
        if (
            b_ox0 is not None
            and b_oy0 is not None
            and b_ox1 is not None
            and b_oy1 is not None
            and b_ox1 > b_ox0
            and b_oy1 > b_oy0
        ):
            rect_mask[b_oy0:b_oy1, b_ox0:b_ox1] = True
            has_drawn = True

    if not has_drawn:
        return None

    if total_bubble_mask is not None:
        rect_mask = np.logical_and(rect_mask, np.logical_not(total_bubble_mask))

    return rect_mask if np.any(rect_mask) else None


def _create_osb_inpainter(config: Settings, verbose: bool):
    """Create the inpainter for the configured OSB method, or None for OpenCV/none."""
    inpainting_method = config.cleaning.inpainter
    inpainter = None
    if inpainting_method == "lama_manga":
        try:
            inpainter = make_inpainter(
                inpainting_method,
                device=config.device,
                luminance_correction=config.cleaning.luminance_correction,
                verbose=verbose,
            )
            log_message("Using LaMa Manga for inpainting", verbose=verbose)
        except Exception as e:
            log_message(
                f"LaMa Manga unavailable ({e}), falling back to OpenCV",
                verbose=verbose,
            )
    if inpainting_method == "none":
        inpainter = None
        log_message(
            "Using text background mode (no inpainting for non-solid regions)",
            verbose=verbose,
        )
    elif inpainting_method == "opencv" or inpainter is None:
        inpainter = None
        log_message("Using OpenCV simple fill for inpainting", verbose=verbose)
    return inpainter


def _border_pixels(
    image: Image.Image, bounds: tuple[int, int, int, int], img_w: int, img_h: int
) -> np.ndarray | None:
    """RGB pixels in an OSB_EXPANSION_PX ring around `bounds`, or None if off-image."""
    x0, y0, x1, y1 = bounds
    sx1 = max(0, x0 - OSB_EXPANSION_PX)
    sy1 = max(0, y0 - OSB_EXPANSION_PX)
    sx2 = min(img_w, x1 + OSB_EXPANSION_PX)
    sy2 = min(img_h, y1 + OSB_EXPANSION_PX)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    crop_np = np.array(image.crop((sx1, sy1, sx2, sy2)).convert("RGB"))
    ring = np.ones(crop_np.shape[:2], dtype=bool)
    ix1 = max(0, x0 - sx1)
    iy1 = max(0, y0 - sy1)
    ix2 = min(crop_np.shape[1], x1 - sx1)
    iy2 = min(crop_np.shape[0], y1 - sy1)
    if ix2 > ix1 and iy2 > iy1:
        ring[iy1:iy2, ix1:ix2] = False
    return crop_np[ring]


def _has_glyph_evidence(glyphs: list[Glyph]) -> bool:
    total = sum(g.area for g in glyphs)
    return len(glyphs) >= 2 or (len(glyphs) == 1 and total >= 30) or total >= 50


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m8)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled.astype(bool)


def _overlap(a: np.ndarray, ax: int, ay: int, b: np.ndarray, bx: int, by: int) -> int:
    """Pixels set in both masks, each placed with its top-left corner at (x, y)."""
    x0, y0 = max(ax, bx), max(ay, by)
    x1 = min(ax + a.shape[1], bx + b.shape[1])
    y1 = min(ay + a.shape[0], by + b.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(
        np.count_nonzero(a[y0 - ay : y1 - ay, x0 - ax : x1 - ax] & b[y0 - by : y1 - by, x0 - bx : x1 - bx])
    )


def _drop_containers(own: list[Glyph], other: list[Glyph]) -> list[Glyph]:
    """Drop glyphs whose interior holds several glyphs and most of the other polarity's area.

    A caption frame or bubble interior is not a glyph. Glyphs that enclose the candidate
    (the frame around an interior) are its border, not its content, so their area is left
    out of the other polarity's total."""
    if not own or not other:
        return own
    total = sum(o.area for o in other)
    kept = []
    for glyph in own:
        h, w = glyph.mask.shape
        nearby = [
            o
            for o in other
            if o.x < glyph.x + w
            and glyph.x < o.x + o.mask.shape[1]
            and o.y < glyph.y + h
            and glyph.y < o.y + o.mask.shape[0]
        ]
        if len(nearby) < 3:
            kept.append(glyph)
            continue
        solid = _fill_holes(glyph.mask)
        inside = enclosing = n_inside = 0
        for o in nearby:
            if _overlap(o.mask, o.x, o.y, solid, glyph.x, glyph.y) >= 0.5 * o.area:
                inside += o.area
                n_inside += 1
            elif _overlap(glyph.mask, glyph.x, glyph.y, _fill_holes(o.mask), o.x, o.y) >= 0.5 * glyph.area:
                enclosing += o.area
        if n_inside < 3 or inside < 0.5 * (total - enclosing):
            kept.append(glyph)
    return kept


def _hollow_fill_area(gray: np.ndarray) -> int:
    """Area of white regions shaped like the fills of hollow (outlined white) letters.

    A fill is a solid, stroke-shaped white region ringed by clearly darker pixels. Compact
    regions (counters), rings around dark letters (white outlines) and interiors holding most
    of the ink (caption frames) are not fills. Unlike the glyph border test, the 1px ring
    still catches outlines too thin or gray to pass it."""
    height, width = gray.shape
    ink = gray < 128
    ink_labels = None
    n, labels, stats, _ = cv2.connectedComponentsWithStats((gray > 200).astype(np.uint8), connectivity=4)
    total = 0
    for label in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < 20 or x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        y0, y1 = max(0, y - 2), min(height, y + h + 2)
        x0, x1 = max(0, x - 2), min(width, x + w + 2)
        m = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        sub = gray[y0:y1, x0:x1]
        ring = cv2.dilate(m, np.ones((3, 3), np.uint8)) & (1 - m)
        fill_median = float(np.median(sub[m == 1]))
        if np.mean(sub[ring == 1] <= fill_median - 80) < 0.6:
            continue
        radius = float(cv2.distanceTransform(m, cv2.DIST_L2, 3).max())
        if area / (4 * radius * radius) < 2.5:
            continue
        solid = _fill_holes(m)
        if solid.sum() - area >= 0.5 * area:
            continue
        _, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = int((hierarchy[0][:, 3] >= 0).sum()) if hierarchy is not None else 0
        if holes > 2:
            if ink_labels is None:
                _, ink_labels = cv2.connectedComponents(ink.astype(np.uint8))
            sub_ink = ink[y0:y1, x0:x1]
            enclosed = int((sub_ink & solid & (m == 0)).sum())
            # Ink just outside the region (its frame, or its own outline) is not its content
            outside = cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool) & ~solid
            border_ids = np.unique(ink_labels[y0:y1, x0:x1][outside & sub_ink])
            border_ids = border_ids[border_ids > 0]
            border = int(np.isin(ink_labels, border_ids).sum()) if border_ids.size else 0
            if enclosed >= 0.5 * max(1, int(ink.sum()) - border):
                continue
        total += area
    return total


def _hole_test_text_color(rgb: np.ndarray) -> tuple[int, int, int]:
    """Text color from an Otsu split: ink enclosing large holes is an outline around the text."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, light = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frame = np.concatenate([light[0], light[-1], light[:, 0], light[:, -1]])
    bg_cls = int(frame.mean() >= 0.5)
    min_area = max(8, gray.size // 2000)

    # Background-class regions not connected to the frame: counters or outlined fills
    _, labels, stats, _ = cv2.connectedComponentsWithStats((light == bg_cls).astype(np.uint8), connectivity=4)
    is_hole = stats[:, cv2.CC_STAT_AREA] >= min_area
    is_hole[0] = False
    is_hole[np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))] = False
    holes = is_hole[labels]
    _, labels, stats, _ = cv2.connectedComponentsWithStats((light != bg_cls).astype(np.uint8), connectivity=8)
    is_ink = stats[:, cv2.CC_STAT_AREA] >= min_area
    is_ink[0] = False
    ink = is_ink[labels]

    use_holes = holes.sum() / max(1, ink.sum()) >= 0.4
    text_mask = holes if use_holes else ink
    text_cls = bg_cls if use_holes else 1 - bg_cls
    core = cv2.erode(text_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pixels = rgb[core] if core.sum() >= 10 else rgb[text_mask]
    if len(pixels):
        median = np.median(pixels, axis=0).astype(int)
        if cv2.cvtColor(np.uint8([[median]]), cv2.COLOR_RGB2HSV)[0][0][1] >= 25:
            return tuple(int(c) for c in median)
    # Snap by Otsu class, not luminance: small JPEG text is often mid-gray
    return (255, 255, 255) if text_cls == 1 else (0, 0, 0)


def _bw_page_text_color(
    pil_image: Image.Image, text_bounds: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    """Text color of an OSB region on a B/W page."""
    x0, y0, x1, y1 = text_bounds
    img_w, img_h = pil_image.size
    pad = 4
    rgb = np.array(
        pil_image.crop(
            (
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(img_w, x1 + pad),
                min(img_h, y1 + pad),
            )
        ).convert("RGB")
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Glyphs touching the padded crop edge are background, not text
    dark = [
        g
        for g in outlined_glyph_components(
            gray, (gray < 50).astype(np.uint8) * 255, lambda border: border > 150
        )
        if not g.on_frame
    ]
    white = [
        g
        for g in outlined_glyph_components(
            gray,
            (gray > 200).astype(np.uint8) * 255,
            lambda border: border < 90,
            min_core_ratio=0.05,
        )
        if not g.on_frame
    ]
    dark, white = _drop_containers(dark, white), _drop_containers(white, dark)
    # Strict borders keep enclosed patches of art from passing as glyphs
    dark = [g for g in dark if g.ratio >= 0.6]
    white = [g for g in white if g.ratio >= 0.6]
    dark_area = sum(g.area for g in dark)
    white_area = sum(g.area for g in white)
    min_area = 0.02 * gray.size
    has_dark = len(dark) >= 2 or dark_area >= min_area
    has_white = len(white) >= 2 or white_area >= min_area

    if has_dark and not has_white:
        return (0, 0, 0)
    if has_white and not has_dark:
        return (255, 255, 255)
    fill_area = _hollow_fill_area(gray)
    if has_dark and has_white:
        return (0, 0, 0) if dark_area >= max(white_area, fill_area) else (255, 255, 255)
    if fill_area >= 0.01 * gray.size:
        return (255, 255, 255)
    return _hole_test_text_color(rgb)


def _color_page_text_color(
    pil_image: Image.Image,
    text_bounds: tuple[int, int, int, int],
    ref_bg_rgb: np.ndarray,
    fallback_fill_color: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    """Text color of an OSB region on a color page."""
    rgb = np.array(pil_image.crop(text_bounds).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Light saturated colors (a yellow banner) clear the gray cutoff but aren't white text
    neutral = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1] <= 40
    dark = outlined_glyph_components(gray, (gray < 50).astype(np.uint8) * 255, lambda border: border > 150)
    white = outlined_glyph_components(
        gray,
        ((gray > 200) & neutral).astype(np.uint8) * 255,
        lambda border: border < 90,
        min_core_ratio=0.05,
    )
    has_dark, has_white = _has_glyph_evidence(dark), _has_glyph_evidence(white)
    if has_dark and not has_white:
        return (0, 0, 0)
    if has_white and not has_dark:
        return (255, 255, 255)
    if has_dark and has_white:
        # By area, not count: dark specks of art outnumber letters; glyphs on the crop
        # edge are mostly background
        dark_area = sum(g.area for g in dark if not g.on_frame)
        white_area = sum(g.area for g in white if not g.on_frame)
        if dark_area != white_area:
            return (0, 0, 0) if dark_area > white_area else (255, 255, 255)
    return _general_text_color(pil_image, text_bounds, ref_bg_rgb, fallback_fill_color)


def _glyph_hue(
    pil_image: Image.Image,
    text_bounds: tuple[int, int, int, int],
    glyph_prob: np.ndarray,
) -> tuple[int, int, int] | None:
    """The hue of letters the text model marks, if most of their cores are colored.

    The map covers whole letters, outlines included, so the colored fill of an outlined
    SFX dominates its cores, while black text keeps them mostly dark."""
    x0, y0, x1, y1 = text_bounds
    img_w, img_h = pil_image.size
    pad = 4
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(img_w, x1 + pad), min(img_h, y1 + pad)
    letters = (glyph_prob[y0:y1, x0:x1] > GLYPH_THRESHOLD).astype(np.uint8)
    core = cv2.erode(letters, np.ones((3, 3), np.uint8)) > 0
    if core.sum() < 30:
        return None
    rgb = np.array(pil_image.crop((x0, y0, x1, y1)).convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colored = core & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 60)
    if colored.sum() < OSB_GLYPH_HUE_MIN_SHARE * core.sum():
        return None
    return tuple(int(c) for c in np.median(rgb[colored], axis=0))


def _detect_osb_text_color(
    pil_image: Image.Image,
    text_bounds: tuple[int, int, int, int],
    ref_bg_rgb: np.ndarray,
    fallback_fill_color: tuple[int, int, int] | None,
    is_color_page: bool,
    glyph_prob: np.ndarray | None = None,
) -> tuple[int, int, int]:
    """Text color of an OSB region, using the detector suited to the page type."""
    if is_color_page:
        # The outline heuristics only tell dark from light, so colored letters with
        # dark outlines would come out black
        hue = None if glyph_prob is None else _glyph_hue(pil_image, text_bounds, glyph_prob)
        if hue is not None:
            return hue
        return _color_page_text_color(pil_image, text_bounds, ref_bg_rgb, fallback_fill_color)
    return _bw_page_text_color(pil_image, text_bounds)


def _general_text_color(
    pil_image: Image.Image,
    text_bounds: tuple[int, int, int, int],
    ref_bg_rgb: np.ndarray,
    fallback_fill_color: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    """Text color of an OSB region: outlined monochrome, then chromatic, then contrast."""
    bg_lab = cv2.cvtColor(np.uint8([[ref_bg_rgb]]), cv2.COLOR_RGB2LAB)[0][0]
    crop_rgb = np.array(pil_image.crop(text_bounds).convert("RGB"))
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    detected_text_color = None

    # Glyph cores whose border contrasts: dark text with a light outline, and the reverse
    dark_glyph_comps = outlined_glyph_components(
        crop_gray,
        (crop_gray < 50).astype(np.uint8) * 255,
        lambda border: border > 150,
    )
    white_glyph_comps = outlined_glyph_components(
        crop_gray,
        (crop_gray > 200).astype(np.uint8) * 255,
        lambda border: border < 90,
        min_core_ratio=0.05,
    )
    has_dark_outlines = _has_glyph_evidence(dark_glyph_comps)
    has_white_outlines = _has_glyph_evidence(white_glyph_comps)

    if has_dark_outlines and not has_white_outlines:
        detected_text_color = (0, 0, 0)
    elif has_white_outlines and not has_dark_outlines:
        detected_text_color = (255, 255, 255)
    elif has_dark_outlines and has_white_outlines:
        avg_dark_ratio = float(np.mean([g.ratio for g in dark_glyph_comps]))
        avg_white_ratio = float(np.mean([g.ratio for g in white_glyph_comps]))
        if len(dark_glyph_comps) > len(white_glyph_comps) and avg_dark_ratio >= avg_white_ratio * 0.8:
            detected_text_color = (0, 0, 0)
        elif len(white_glyph_comps) > len(dark_glyph_comps) and avg_white_ratio >= avg_dark_ratio * 0.8:
            detected_text_color = (255, 255, 255)
        elif avg_dark_ratio > avg_white_ratio:
            detected_text_color = (0, 0, 0)
        else:
            detected_text_color = (255, 255, 255)

    # Chromatic text: LAB contrast against the background
    if detected_text_color is None:
        crop_lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        dist_map = np.linalg.norm(crop_lab - bg_lab.astype(np.float32), axis=2)

        robust_max_dist = np.percentile(dist_map, 95)
        CONTRAST_THRESHOLD = max(30, robust_max_dist * 0.6)
        contrast_mask = (dist_map > CONTRAST_THRESHOLD).astype(np.uint8) * 255

        kernel_3 = np.ones((3, 3), np.uint8)
        contrast_mask = cv2.morphologyEx(contrast_mask, cv2.MORPH_CLOSE, kernel_3)
        contrast_mask = cv2.erode(
            contrast_mask,
            np.ones((2, 2), np.uint8),
            iterations=1,
        )

        contours, _ = cv2.findContours(
            contrast_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        clean_mask = np.zeros_like(contrast_mask)
        MIN_COMPONENT_AREA = 4
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_COMPONENT_AREA:
                cv2.drawContours(
                    clean_mask,
                    [cnt],
                    -1,
                    255,
                    cv2.FILLED,
                )

        text_pixels_rgb = crop_rgb[clean_mask == 255]
        if len(text_pixels_rgb) >= 10:
            sampled_rgb = tuple(np.median(text_pixels_rgb, axis=0).astype(int))
            hsv = cv2.cvtColor(
                np.uint8([[sampled_rgb]]),
                cv2.COLOR_RGB2HSV,
            )[0][0]
            if hsv[1] >= 25:
                detected_text_color = sampled_rgb

    # Non-outlined monochrome text: contrast against the fill
    if detected_text_color is None:
        fill_for_contrast = fallback_fill_color if fallback_fill_color is not None else (0, 0, 0)
        fill_lum = 0.299 * fill_for_contrast[0] + 0.587 * fill_for_contrast[1] + 0.114 * fill_for_contrast[2]
        detected_text_color = (0, 0, 0) if fill_lum >= 128 else (255, 255, 255)
    return detected_text_color


class _RegionFill(NamedTuple):
    """How to fill one OSB region, from `_decide_osb_region_fill`."""

    composite_clip_bbox: tuple[int, int, int, int] | None
    original_bounds: tuple[int | None, int | None, int | None, int | None]
    fill_color: tuple[int, int, int] | None  # None: the region needs inpainting
    fallback_fill_color: tuple[int, int, int] | None
    text_color: tuple[int, int, int] | None


def _decide_osb_region_fill(
    work: OutsideTextWork,
    current_image: Image.Image,
    group: dict[str, Any],
    region_number: int,
) -> _RegionFill:
    """Sample an OSB region's background and text colors to choose how to fill it."""
    img_w, img_h = work.img_w, work.img_h
    verbose = work.verbose
    composite_clip_bbox = None
    fill_color = None
    fallback_fill_color = None
    text_color = None
    bounds = (None, None, None, None)
    original_bbox_dict = group.get("original_bbox")
    if not original_bbox_dict:
        return _RegionFill(composite_clip_bbox, bounds, fill_color, fallback_fill_color, text_color)
    ox = int(original_bbox_dict.get("x", 0))
    oy = int(original_bbox_dict.get("y", 0))
    ow = int(original_bbox_dict.get("width", 0))
    oh = int(original_bbox_dict.get("height", 0))
    if ow <= 0 or oh <= 0:
        return _RegionFill(composite_clip_bbox, bounds, fill_color, fallback_fill_color, text_color)
    bounds = (
        max(0, min(img_w, ox)),
        max(0, min(img_h, oy)),
        max(0, min(img_w, ox + ow)),
        max(0, min(img_h, oy + oh)),
    )
    composite_clip_bbox = (ox, oy, ox + ow, oy + oh)

    # Determine detected text color for this region to ensure contrast
    if work.original_text_colors:
        votes_dark = 0
        votes_light = 0
        gx1, gy1, gx2, gy2 = composite_clip_bbox
        for (bx1, by1, bx2, by2), t_dark in work.original_text_colors.items():
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2
            if gx1 <= bcx <= gx2 and gy1 <= bcy <= gy2:
                if t_dark:
                    votes_dark += 1
                else:
                    votes_light += 1
        if votes_dark > 0 or votes_light > 0:
            group_bg_is_dark = votes_dark >= votes_light

            # Detected value represents background brightness
            fallback_fill_color = (0, 0, 0) if group_bg_is_dark else (255, 255, 255)

            t_type = "Dark" if group_bg_is_dark else "Light"
            f_col = "White" if fallback_fill_color == (255, 255, 255) else "Black"
            log_message(
                f"OSB Region {region_number}: Detected {t_type} background. Fallback fill: {f_col}.",
                verbose=verbose,
            )

    # Expanded sampling around the original bbox to find background color
    mask_indices = group.get("mask_indices", [])
    if mask_indices and work.raw_outside_text_results:
        text_bounds = results_bounds(work.raw_outside_text_results, mask_indices)
    else:
        text_bounds = composite_clip_bbox
    border_pixels = _border_pixels(current_image, text_bounds, img_w, img_h)
    if border_pixels is None or len(border_pixels) < 20:
        return _RegionFill(composite_clip_bbox, bounds, fill_color, fallback_fill_color, text_color)

    detected_color, solid_ratio = border_color(border_pixels)
    is_border_solid = solid_ratio >= OSB_SOLID_RATIO_THRESHOLD
    if is_border_solid or fallback_fill_color is None:
        fallback_fill_color = detected_color

    ref_bg_rgb = np.array(
        detected_color
        if is_border_solid
        else (fallback_fill_color if fallback_fill_color is not None else (0, 0, 0)),
        dtype=np.uint8,
    )
    text_color = _detect_osb_text_color(
        work.pil_image,
        text_bounds,
        ref_bg_rgb,
        fallback_fill_color,
        work.is_color_page,
        work.glyph_map.get() if work.is_color_page else None,
    )

    force_fill = work.config.cleaning.inpainter == "opencv"

    # Get expanded bounds for this group to check solid color and for cv2 fill
    p_x0, p_y0, p_x1, p_y1 = composite_clip_bbox
    if mask_indices and work.outside_text_results:
        p_x0, p_y0, p_x1, p_y1 = results_bounds(work.outside_text_results, mask_indices)
        p_x0, p_y0 = max(0, p_x0), max(0, p_y0)
        p_x1, p_y1 = min(img_w, p_x1), min(img_h, p_y1)

    expanded_is_solid = is_border_solid
    if not force_fill:
        eborder_pixels = _border_pixels(current_image, (p_x0, p_y0, p_x1, p_y1), img_w, img_h)
        if eborder_pixels is not None and eborder_pixels.size > 0:
            edetected_color, esolid_ratio = border_color(eborder_pixels)
            if esolid_ratio >= OSB_SOLID_RATIO_THRESHOLD:
                expanded_is_solid = True
                fallback_fill_color = edetected_color

    if expanded_is_solid or force_fill:
        fill_color = fallback_fill_color if fallback_fill_color is not None else detected_color
        if force_fill:
            log_message(
                f"Using OpenCV simple fill for OSB region: {fill_color} background",
                verbose=verbose,
            )
        else:
            log_message(
                f"Skipping inpainting for OSB region: detected solid {fill_color} background",
                verbose=verbose,
            )

    return _RegionFill(composite_clip_bbox, bounds, fill_color, fallback_fill_color, text_color)


def _clip_mask_to_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    """Copy of `mask` with everything outside `bbox` cleared; a plain copy without one."""
    if bbox is None:
        return mask.copy()
    img_h, img_w = mask.shape
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, min(img_w, x1)), max(0, min(img_w, x2))
    y1, y2 = max(0, min(img_h, y1)), max(0, min(img_h, y2))
    clipped = np.zeros_like(mask)
    if x2 > x1 and y2 > y1:
        clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def _glyph_inpaint_mask(
    work: OutsideTextWork,
    combined_mask: np.ndarray,
    composite_clip_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """The region's mask clipped to its text box, then shrunk to the dilated glyphs,
    including those found in the box at twice the text scale."""
    return work.glyph_map.lama_mask(_clip_mask_to_bbox(combined_mask, composite_clip_bbox))


def _try_model_inpaint(
    inpainter,
    image: Image.Image,
    mask: np.ndarray,
    seed: int,
    composite_clip_bbox,
    verbose: bool,
) -> tuple[Image.Image | None, str | None]:
    """Inpaint one OSB region with the inpainting model; returns (image, None) or (None, failure reason)."""
    if inpainter is None:
        return None, "inpainter unavailable"
    try:
        inpainted_image = inpainter.inpaint_mask(
            image,
            mask,
            seed=seed,
            verbose=verbose,
            strict_mask_clipping=True,
            composite_clip_bbox=composite_clip_bbox,
        )
    except CancellationError:
        raise
    except Exception as e:
        return None, f"inpainting error: {e}"
    if inpainted_image is image:
        return None, "inpainter returned the original image (no inpaint)"
    return inpainted_image, None


def _run_inpaint_waves(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    inpainter,
    request_coordinator,
    fill_candidate,
    composite_candidate,
    verbose: bool,
) -> tuple[Image.Image, int, int]:
    """Run queued per-region inpainting jobs in non-overlapping waves via the coordinator.

    Each result goes through `composite_candidate(before, after, candidate)`; regions
    whose job fails get `fill_candidate(image, candidate, color)` instead.
    Returns the image and the number of regions done by the model and by CV2.
    """
    model_count = 0
    cv2_count = 0
    waves = partition_non_overlapping_waves(
        candidates,
        lambda candidate: candidate["context_bbox"],
    )
    log_message(
        f"Scheduling OSB inpainting in {len(waves)} wave(s)",
        verbose=verbose,
    )

    for wave in waves:
        base_image = image

        def make_job(candidate, base_img=base_image):
            def job():
                image_for_job = base_img.copy()
                try:
                    result_image = inpainter.inpaint_mask(
                        image_for_job,
                        candidate["mask"],
                        seed=candidate["seed"],
                        verbose=verbose,
                        strict_mask_clipping=True,
                        composite_clip_bbox=candidate["composite_clip_bbox"],
                    )
                    if result_image is image_for_job:
                        raise RuntimeError("inpainter returned the original image (no inpaint)")
                    return {
                        "candidate": candidate,
                        "image": result_image,
                        "error": None,
                    }
                except CancellationError:
                    raise
                except Exception as e:
                    return {
                        "candidate": candidate,
                        "image": None,
                        "error": e,
                    }

            return job

        results = request_coordinator.map_ordered([make_job(candidate) for candidate in wave])
        for result in results:
            candidate = result["candidate"]
            if result["error"] is not None:
                fallback_color_to_use = candidate["fallback_color"]
                log_message(
                    f"Inpainting failed for OSB region {candidate['index']}"
                    f" (inpainting error: {result['error']}); "
                    f"falling back to CV2 fill ({fallback_color_to_use})",
                    always_print=True,
                )
                image = fill_candidate(image, candidate, fallback_color_to_use)
                cv2_count += 1
                continue

            result_image = composite_candidate(base_image, result["image"], candidate)
            context_bbox = candidate["context_bbox"]
            if context_bbox is None:
                image = result_image
            else:
                image = paste_image_region(
                    image,
                    result_image,
                    context_bbox,
                )
            model_count += 1
    return image, model_count, cv2_count


def finish_outside_text_work(
    work: OutsideTextWork,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Run OSB inpainting for prepared work and attach render metadata."""
    pil_image = work.pil_image
    config = work.config
    verbose = work.verbose
    outside_text_results = work.outside_text_results
    total_bubble_mask = work.total_bubble_mask
    mask_groups = work.mask_groups
    img_w = work.img_w
    img_h = work.img_h
    outside_text_data = work.outside_text_data

    extracted_text_colors = {}
    none_skipped_clip_bboxes = set()
    current_image = pil_image

    try:
        log_message("Inpainting outside text regions...", verbose=verbose)

        inpainting_method = config.cleaning.inpainter
        inpainter = _create_osb_inpainter(config, verbose)
        current_image = pil_image
        none_skipped_clip_bboxes = set()
        if mask_groups:
            base_seed = random.randint(1, 999999) if config.cleaning.seed == -1 else config.cleaning.seed

            extracted_text_colors = {}
            model_inpaints = 0
            cv2_inpaints = 0
            none_skips = 0
            request_coordinator = config.request_coordinator
            pending_inpaints = []

            def simple_fill(image, color, group, combined_mask, original_bbox_dict, bounds):
                """Fill the group's (tilt-aware) text boxes, else its whole mask.

                Boxes whose text gets rendered are left to the renderer, which paints
                them at the angle the translation is drawn at.
                """
                mask_indices = group.get("mask_indices", [])
                rect_mask = _build_osb_simple_fill_mask(
                    img_w=img_w,
                    img_h=img_h,
                    mask_indices=mask_indices,
                    outside_text_results=outside_text_results,
                    outside_text_data=outside_text_data,
                    config=config,
                    fallback_bounds=bounds if original_bbox_dict else None,
                    total_bubble_mask=total_bubble_mask,
                )
                boxes = _osb_fill_boxes(mask_indices, outside_text_results, img_w, img_h)
                if rect_mask is not None and boxes and all(idx < len(outside_text_data) for idx, _ in boxes):
                    for idx, box in boxes:
                        item = outside_text_data[idx]
                        item["pending_fill"] = _PendingOsbFill(
                            color,
                            box,
                            _effective_osb_tilt(item, config),
                            total_bubble_mask,
                        )
                    return image
                return paint_mask(image, combined_mask if rect_mask is None else rect_mask, color)

            def apply_candidate_simple_fill(image, candidate, color_to_use):
                return simple_fill(
                    image,
                    color_to_use,
                    candidate["group"],
                    candidate["fill_mask"],
                    candidate["original_bbox_dict"],
                    candidate["original_bounds"],
                )

            def flush_pending_inpaints():
                nonlocal current_image, model_inpaints, cv2_inpaints
                if not pending_inpaints:
                    return

                candidates = list(pending_inpaints)
                pending_inpaints.clear()
                current_image, model_done, cv2_done = _run_inpaint_waves(
                    current_image,
                    candidates,
                    inpainter,
                    request_coordinator,
                    apply_candidate_simple_fill,
                    lambda before, after, candidate: after,
                    verbose,
                )
                model_inpaints += model_done
                cv2_inpaints += cv2_done

            for i, group in enumerate(mask_groups):
                raise_if_cancelled()
                log_message(
                    f"Inpainting outside text region {i + 1}/{len(mask_groups)}",
                    verbose=verbose,
                )
                combined_mask = group["combined_mask"]
                combined_mask = np.logical_and(combined_mask, np.logical_not(total_bubble_mask))
                if not np.any(combined_mask):
                    log_message(
                        "Skipping outside text region after bubble masking (no remaining area)",
                        verbose=verbose,
                    )
                    continue
                region_seed = base_seed + i if base_seed > 0 else base_seed

                original_bbox_dict = group.get("original_bbox")
                region = _decide_osb_region_fill(work, current_image, group, region_number=i + 1)
                composite_clip_bbox = region.composite_clip_bbox
                fill_color = region.fill_color
                fallback_fill_color = region.fallback_fill_color
                if region.text_color is not None:
                    extracted_text_colors[composite_clip_bbox] = region.text_color

                region_fill_args = (
                    group,
                    combined_mask,
                    original_bbox_dict,
                    region.original_bounds,
                )

                if fill_color is not None:
                    flush_pending_inpaints()
                    current_image = simple_fill(current_image, fill_color, *region_fill_args)
                    cv2_inpaints += 1
                    continue

                if inpainting_method == "none":
                    if composite_clip_bbox:
                        none_skipped_clip_bboxes.add(composite_clip_bbox)
                    none_skips += 1
                    log_message(
                        f"Skipping inpaint for non-solid OSB region {i + 1} (none mode)",
                        verbose=verbose,
                    )
                    continue

                # LaMa erases every masked pixel, so give it only the letters
                inpaint_mask = combined_mask
                box_mask = _clip_mask_to_bbox(combined_mask, composite_clip_bbox)
                if inpainting_method == "lama_manga" and inpainter is not None:
                    inpaint_mask = _glyph_inpaint_mask(work, combined_mask, composite_clip_bbox)

                if request_coordinator is not None and inpainter is not None:
                    fallback_color_to_use = fallback_fill_color if fallback_fill_color else (255, 255, 255)
                    pending_inpaints.append(
                        {
                            "index": i + 1,
                            "mask": inpaint_mask.copy(),
                            "fill_mask": combined_mask.copy(),
                            "box_mask": box_mask,
                            "seed": region_seed,
                            "composite_clip_bbox": composite_clip_bbox,
                            "fallback_color": fallback_color_to_use,
                            "group": group,
                            "original_bbox_dict": original_bbox_dict,
                            "original_bounds": region.original_bounds,
                            "context_bbox": expanded_mask_bbox(inpaint_mask, current_image.size),
                        }
                    )
                    log_message(
                        f"Queued OSB region {i + 1} for intra-page inpaint scheduling",
                        verbose=verbose,
                    )
                    continue

                inpainted_image, inpaint_fail_reason = _try_model_inpaint(
                    inpainter,
                    current_image,
                    inpaint_mask,
                    region_seed,
                    composite_clip_bbox,
                    verbose,
                )

                if inpaint_fail_reason is not None:
                    fallback_color_to_use = fallback_fill_color if fallback_fill_color else (255, 255, 255)
                    log_message(
                        f"Inpainting failed for OSB region {i + 1} ({inpaint_fail_reason})"
                        f"; falling back to CV2 fill ({fallback_color_to_use})",
                        always_print=True,
                    )
                    current_image = simple_fill(current_image, fallback_color_to_use, *region_fill_args)
                    cv2_inpaints += 1
                    continue

                model_inpaints += 1
                current_image = inpainted_image

            flush_pending_inpaints()

            log_message("Outside text inpainting completed", verbose=verbose)
            parts = [
                f"Model: {model_inpaints}",
                f"CV2: {cv2_inpaints}",
            ]
            if none_skips:
                parts.append(f"Skipped (none): {none_skips}")
            log_message(
                f"Inpainted {len(mask_groups)} outside text regions ({', '.join(parts)})",
                always_print=True,
            )

        apply_inpaint_render_metadata(
            outside_text_data,
            extracted_text_colors,
            none_skipped_clip_bboxes,
        )
        return current_image, outside_text_data

    except CancellationError:
        raise
    except Exception as e:
        log_message(
            f"Error during outside text inpainting: {e}",
            always_print=True,
        )
        return pil_image, outside_text_data


def process_outside_text(
    pil_image: Image.Image,
    config: Settings,
    image_path: str | Path,
    image_format: str | None,
    verbose: bool = False,
    bubble_data: list[dict[str, Any]] | None = None,
    text_free_boxes: list[list[float]] | None = None,
    panels: list[tuple[int, int, int, int]] | None = None,
    glyph_map: PageGlyphMap | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """
    Process outside text detection, inpainting, and prepare data for translation.

    This function handles the complete outside text processing pipeline:
    1. Detects text outside speech bubbles using OCR
    2. Inpaints the detected text regions
    3. Prepares the outside text data for translation API calls

    Args:
        pil_image: The PIL image to process
        config: Settings containing all settings
        image_path: Path to the original image file
        image_format: Original image format (PNG, JPEG, etc.)
        processing_scale: The scale factor for image processing
        verbose: Whether to print detailed logging
        glyph_map: The page's text map, shared with bubble cleaning

    Returns:
        Tuple containing:
        - processed_pil_image: The image after outside text inpainting
        - outside_text_data: List of dicts with outside text information for translation
    """
    work = prepare_outside_text_work(
        pil_image,
        config,
        image_path,
        image_format,
        verbose=verbose,
        bubble_data=bubble_data,
        text_free_boxes=text_free_boxes,
        panels=panels,
        glyph_map=glyph_map,
    )
    if work is None:
        return pil_image, []
    return finish_outside_text_work(work)
