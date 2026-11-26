"""Images and masks in and out of a project file's blob table.

Everything is stored as PNG, optimised by oxipng (lossless; it also stores an RGBA page that is
really gray as gray). Masks are 1-bit PNGs of their bounding crop (an empty mask as a 1x1 crop),
so a full-page mask with a small bubble in it costs a few hundred bytes. Edits to a page are
patches of only the pixels that changed (diff_patches).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import oxipng
from PIL import Image

from mangatl.project.model import ImageRef, MaskRef, Patch

if TYPE_CHECKING:
    from mangatl.project.store import Project


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)  # oxipng recompresses anyway
    return oxipng.optimize_from_memory(buf.getvalue(), level=2)


def _open(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def put_mask(project: Project, mask: np.ndarray) -> MaskRef:
    """Store a binary mask (bool, or uint8 holding 0 and one other value)."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {mask.shape}")
    values = np.unique(mask)
    if mask.dtype == np.bool_:
        dtype, on = "bool", 1
    elif mask.dtype == np.uint8 and len(values) <= 2 and (len(values) < 2 or values[0] == 0):
        dtype, on = "uint8", int(values[-1]) if len(values) and values[-1] else 255
    else:
        raise ValueError(f"not a binary mask: {mask.dtype} {values[:5]}")
    page_h, page_w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys):
        x, y, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    else:
        x, y, x2, y2 = 0, 0, 1, 1
    crop = Image.fromarray(mask[y:y2, x:x2] != 0).convert("1")
    blob = project.put_blob(_png(crop))
    return MaskRef(blob=blob, x=x, y=y, w=x2 - x, h=y2 - y, page_w=page_w, page_h=page_h, dtype=dtype, on=on)


def get_mask(project: Project, ref: MaskRef) -> np.ndarray:
    crop = np.asarray(_open(project.get_blob(ref.blob)).convert("1"), dtype=bool)
    out = np.zeros((ref.page_h, ref.page_w), dtype=bool)
    out[ref.y : ref.y + ref.h, ref.x : ref.x + ref.w] = crop
    if ref.dtype == "bool":
        return out
    return out.astype(np.uint8) * np.uint8(ref.on)


def put_image(project: Project, image: Image.Image) -> ImageRef:
    return ImageRef(blob=project.put_blob(_png(image)), mode=image.mode, w=image.width, h=image.height)


def get_image(project: Project, ref: ImageRef) -> Image.Image:
    image = _open(project.get_blob(ref.blob))
    return image if image.mode == ref.mode else image.convert(ref.mode)


def diff_patches(project: Project, base: Image.Image, edited: Image.Image, tile: int = 64) -> list[Patch]:
    """Patches that turn `base` into `edited` (same size and mode): per band of `tile` rows, the
    changed pixels' bounding crop, holding only the changed pixels (the rest blank, under a
    1-bit mask)."""
    if base.mode not in ("L", "RGB", "RGBA"):
        raise ValueError(f"unsupported mode {base.mode}")
    a, b = np.asarray(base), np.asarray(edited)
    if a.shape != b.shape or base.mode != edited.mode:
        raise ValueError(f"size/mode differ: {base.mode}{a.shape} vs {edited.mode}{b.shape}")
    changed = a != b
    if changed.ndim == 3:
        changed = changed.any(axis=2)
    patches = []
    for top in range(0, changed.shape[0], tile):
        band = changed[top : top + tile]
        ys, xs = np.nonzero(band)
        if not len(ys):
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = top + int(ys.min()), top + int(ys.max()) + 1
        keep = changed[y0:y1, x0:x1]
        pixels = b[y0:y1, x0:x1].copy()
        # blank in a changed pixel's color: a gray, opaque edit stays a gray, opaque PNG
        pixels[~keep] = pixels[keep][0]
        mask = None
        if not keep.all():
            mask = project.put_blob(_png(Image.fromarray(keep).convert("1")))
        crop = Image.fromarray(pixels)
        if crop.mode != edited.mode:
            crop = crop.convert(edited.mode)
        patches.append(Patch(blob=project.put_blob(_png(crop)), x=x0, y=y0, mask=mask))
    return patches


def apply_patches(project: Project, base: Image.Image, patches: list[Patch]) -> Image.Image:
    out = base.copy()
    for p in patches:
        crop = _open(project.get_blob(p.blob))
        crop = crop if crop.mode == out.mode else crop.convert(out.mode)
        mask = _open(project.get_blob(p.mask)).convert("1") if p.mask else None
        out.paste(crop, (p.x, p.y), mask)
    return out
