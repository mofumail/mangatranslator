"""Render a page from the project file alone: what the editor will do after every edit, and the
proof (eval/rebuild_check.py) that a project holds everything its translated pages are made of."""

from __future__ import annotations

from PIL import Image

from mangatl.project.capture import canvas_image, render_inputs
from mangatl.project.store import Project


def _as_mode(image: Image.Image, mode: str) -> Image.Image:
    return image if image.mode == mode else image.convert(mode)


def rebuild_page(project: Project, page_id: str) -> Image.Image:
    """The page as the pipeline returned it (saved pages: converted to the saved mode)."""
    from mangatl.clean.outside_text import paint_pending_osb_fills
    from mangatl.typeset.page import render_translations

    page = project.load_page(page_id)
    if page.sections:  # a long strip: its sections, stitched
        parts = [_as_mode(rebuild_page(project, s), page.target_mode) for s in page.sections]
        image = Image.new(page.target_mode, (parts[0].width, sum(p.height for p in parts)))
        top = 0
        for part in parts:
            image.paste(part, (0, top))
            top += part.height
        return image
    if page.render is None:
        image = canvas_image(project, page)
    else:
        x = render_inputs(project, page)
        image = render_translations(
            x.canvas,
            x.sorted_bubble_data,
            x.translated_texts,
            x.processed_bubbles_info,
            x.page_cv,
            x.config,
            x.sizes,
            x.processing_scale,
            [],
            None,
            x.page_glyphs,
        )
        image = paint_pending_osb_fills(image, x.outside_text_data)
    return _as_mode(image, page.target_mode) if page.output else image
