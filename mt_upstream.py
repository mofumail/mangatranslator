"""Run MangaTranslator's CLI (MangaTranslator/main.py) with local patches installed.

Used by translate.py; takes exactly the same arguments as MangaTranslator/main.py.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "MangaTranslator"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(1, str(ROOT))

import osb_ocr  # noqa: E402

osb_ocr.install()
osb_ocr.install_flat_fill_check()
osb_ocr.install_narration_style()
osb_ocr.install_keep_line_breaks()
osb_ocr.install_wrapped_paragraphs()
argv = sys.argv[1:]
vision_ocr = "--ocr-method" in argv and argv[argv.index("--ocr-method") + 1] == "LLM"
if not vision_ocr:  # keep LLM vision crops small; local OCR needs full resolution
    osb_ocr.install_no_downscale()
sys.argv[0] = str(UPSTREAM / "main.py")
runpy.run_path(sys.argv[0], run_name="__main__")
