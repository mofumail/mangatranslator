"""Run the pipeline CLI (mangatl/cli.py) with the local patches installed.

Used by translate.py; takes exactly the same arguments as mangatl/cli.py.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import osb_ocr  # noqa: E402

osb_ocr.install_page_image()
osb_ocr.install_truncation_guard()
osb_ocr.install_keep_inner_quotes()
osb_ocr.install()
osb_ocr.install_flat_fill_check()
osb_ocr.install_narration_style()
osb_ocr.install_symbol_fallback()
osb_ocr.install_glyph_substitutes()
osb_ocr.install_no_empty_bubbles()
osb_ocr.install_keep_line_breaks()
osb_ocr.install_wrapped_paragraphs()
osb_ocr.install_emphasis_sizes()
osb_ocr.install_layout_preferences()
osb_ocr.install_min_line_height()
osb_ocr.install_vertical_only_if_much_bigger()
osb_ocr.install_keep_junk_osb()
osb_ocr.install_sfx_labels()
osb_ocr.install_vision_short()
osb_ocr.install_character_notes()
osb_ocr.install_text_dump()
osb_ocr.install_llm_cache()
argv = sys.argv[1:]
vision_ocr = "--ocr-method" in argv and argv[argv.index("--ocr-method") + 1] == "LLM"
if not vision_ocr:  # keep LLM vision crops small; local OCR needs full resolution
    osb_ocr.install_no_downscale()
sys.argv[0] = str(ROOT / "mangatl" / "cli.py")
runpy.run_module("mangatl.cli", run_name="__main__", alter_sys=True)
