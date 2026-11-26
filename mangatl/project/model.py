"""The page document: everything known about one page, shared by the batch pipeline, the editor and
the agent.

A page holds its images (by blob reference into the project file) and its regions. A region is
one piece of text (a speech bubble or text on the art) with a stable, never reused id; its
position in reading order is a separate field, so the numbers shown to a user (1, 2, 3, ...) may
change while the region underneath stays the same. Each region carries the results of every
stage (OCR read, translation, cleaning, the inputs typesetting needs) with a status per stage.

Images and masks are stored as blobs in the project file (mangatl.project.store); here they are
references: content hashes, plus the placement of a mask crop on the page.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Box = tuple[int, int, int, int]
Color = tuple[int, ...]  # RGB / BGR, sometimes with alpha (cleaning works on BGRA pages)
STAGES = ("detect", "ocr", "translate", "clean", "typeset")
Status = Literal["pending", "done", "stale", "user", "error"]


class _Doc(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MaskRef(_Doc):
    """A binary mask stored as a 1-bit PNG of its bounding crop, placed at (x, y) on a page of
    page_w x page_h. Rebuilt as an array of `dtype` holding 0 and `on`."""

    blob: str  # sha256 of the PNG
    x: int
    y: int
    w: int
    h: int
    page_w: int
    page_h: int
    dtype: Literal["uint8", "bool"] = "uint8"
    on: int = 255


class ImageRef(_Doc):
    blob: str  # sha256 of the PNG (or of the source file's bytes)
    mode: str
    w: int
    h: int


class Patch(_Doc):
    """Pixels that differ from a base image: a PNG crop pasted at (x, y) through `mask` (a 1-bit
    PNG of the pixels that changed; the crop's other pixels are blank, so they cost nothing)."""

    blob: str
    x: int
    y: int
    mask: str | None = None  # None: the whole crop


class OutsideText(_Doc):
    """What text on the art (outside bubbles) carries beyond a bubble's fields."""

    original_bbox: Box  # the detection, before expansion for rendering
    dark_text: bool
    text_color: Color | None = None
    aspect_ratio: float
    needs_background: bool
    tilt_deg: float
    tilt_conf: float
    tilt_axis: str
    oriented_size: tuple[float, float]
    detect_index: int  # position in detection order (pending fills are painted in this order)
    narration: bool | None = None  # narration / dialogue on the art (typeset.narration)
    narration_text: str | None = None


class PendingFill(_Doc):
    """A solid fill for outside text, painted at the angle its translation is drawn at."""

    color: Color
    box: Box
    tilt_deg: float
    bubble_mask: MaskRef | None = None  # areas the fill must not cover (bubbles)


class BubbleCleaning(_Doc):
    """How a speech bubble was cleaned; typesetting draws inside `mask`."""

    bbox: Box
    mask: MaskRef | None = None
    base_mask: MaskRef | None = None
    art_mask: MaskRef | None = None
    color: Color
    is_colored: bool
    text_bbox: Box | None = None
    text_color_bgr: Color | None = None
    is_seg: bool
    inpainted: bool


class OriginalCrop(_Doc):
    """The original lettering under a region (restored when the text is kept as drawn): cut from
    the work image in `mode`, or stored as `image` when it can't be (taken before cleaning)."""

    mode: str
    image: ImageRef | None = None


class Region(_Doc):
    id: str  # stable, never reused within the page
    order: int | None  # position in reading order; None: outside text that isn't translated
    kind: Literal["bubble", "outside"]
    bbox: Box
    confidence: float
    label: str | None = None  # detector class
    panel: int | None = None
    seg_mask: MaskRef | None = None
    neighbors: list[Box] | None = None  # conjoined bubbles next to this one
    ocr: str | None = None
    translation: str | None = None
    skip: bool = False  # erased, not translated (e.g. a Chinese block beside the Japanese)
    outside: OutsideText | None = None
    cleaning: BubbleCleaning | None = None
    pending_fill: PendingFill | None = None
    original: OriginalCrop | None = None
    status: dict[str, Status] = Field(default_factory=dict)
    locked: bool = False  # the pipeline never changes a locked region
    flag: str | None = None  # "vanished": re-detection no longer finds it (kept for its edits)


class GlyphSize(_Doc):
    bbox: Box
    size: float
    parts: int


class Glyphs(_Doc):
    """The source lettering's size: the page's typical size and each bubble's own."""

    page: float | None
    bubbles: list[GlyphSize] = Field(default_factory=list)


Num = int | float


class RenderState(_Doc):
    """Page-level typesetting inputs, measured once (and later overridable by the user)."""

    processing_scale: float
    main_min_font: Num
    main_max_font: Num
    osb_min_font: Num
    osb_max_font: Num
    padding_pixels: Num
    osb_padding_pixels: Num
    osb_outline_ratio: Num
    glyphs: Glyphs | None = None  # None: not measured (auto font size off)


class SourceRef(_Doc):
    path: str  # where it was read from
    blob: str  # the file's bytes, as read
    w: int
    h: int
    crop: Box | None = None  # a strip's section: this part of the file's image...
    crop_mode: str | None = None  # ...decoded in this mode


class Page(_Doc):
    id: str  # the source path relative to the run's input folder ("<id>#<n>": a strip's section)
    source: SourceRef
    work_mode: str  # the source converted to this mode is the base of the work image
    work_patches: list[Patch] = Field(default_factory=list)  # ...plus these (text on the art erased)
    work_image: ImageRef | None = None  # the whole work image, when it isn't the source's size
    canvas_mode: str
    cleaned: list[Patch] = Field(default_factory=list)  # the canvas text is drawn on, over the work image
    target_mode: str  # mode of the saved result
    regions: list[Region] = Field(default_factory=list)
    next_region: int = 1
    panels: list[Box] | None = None  # comic panels (reading order); None: not detected
    render: RenderState | None = None  # None: nothing typeset; the canvas is the result
    sections: list[str] = Field(default_factory=list)  # a long strip: the pages of its sections
    settings: str  # id of the settings document the page was made with
    output: str | None = None  # where the translated page was written
    version: int = 1

    def region(self, region_id: str) -> Region:
        for r in self.regions:
            if r.id == region_id:
                return r
        raise KeyError(region_id)
