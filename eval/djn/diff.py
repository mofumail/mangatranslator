"""Side-by-side OCR + translation of two doujin runs: diff.py <label a> <label b> [page substring]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bake"))
import sheet
out = Path(__file__).resolve().parent / "out"
a, b = (sheet.parse(out / f"{l}.log") for l in sys.argv[1:3])
only = sys.argv[3] if len(sys.argv) > 3 else ""
for page in a:
    if only not in page: continue
    print(f"=== {page}")
    for tag, run in ((sys.argv[1], a), (sys.argv[2], b)):
        print(f"  [{tag}]")
        for src, tr in run.get(page, []):
            print(f"    {src[:60]!r:64} -> {tr[:70]!r}")
