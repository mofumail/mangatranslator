"""What the short-item vision pass changed in one run: vision_changes.py <run.log> <text dump dir> [book]

Prints, per changed item: page, Japanese, the text-only draft, the vision answer, and (for an
OpenMantra book) the pro English of the box. Drafts come from the same run, so this is an exact
A/B of the vision pass.
"""
import json
import re
import sys
from pathlib import Path

log, dump = Path(sys.argv[1]), Path(sys.argv[2])
book = sys.argv[3] if len(sys.argv) > 3 else None
ann = {}
if book:
    def cover(a, b):  # share of box a covered by box b
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / max(1, (a[2] - a[0]) * (a[3] - a[1]))
    data = json.loads((Path(__file__).resolve().parent / "openmantra" / "annotation.json").read_text())
    pages = next(b for b in data if b["book_title"] == book)["pages"]
    ann = {Path(p["image_paths"]["ja"]).stem: p["text"] for p in pages}
page = None
for line in log.read_text(errors="replace").splitlines():
    m = re.match(r"Processing \d+/\d+: (.+?)\.\w+\b", line)
    if m:
        page = m.group(1)
        continue
    m = re.search(r"Vision, item (\d+): (.*) -> (.*)$", line)
    if not m or not page:
        continue
    i = int(m.group(1)) - 1
    items = json.loads((dump / f"{page}.json").read_text())
    it = items[i] if i < len(items) else {}
    pro = ""
    if book and it.get("bbox"):
        best = max(((cover((t["x"], t["y"], t["x"] + t["w"], t["y"] + t["h"]), it["bbox"]), t)
                    for t in ann.get(page, [])), key=lambda c: c[0], default=(0, None))
        pro = best[1]["text_en"].replace("\n", " ") if best[0] >= 0.3 else "(no pro box)"
    print(f"{page} #{i + 1} {'OSB' if it.get('osb') else 'BUB'} {it.get('ocr', '')!r}")
    print(f"   text  : {m.group(2)}\n   vision: {m.group(3)}" + (f"\n   pro   : {pro}" if book else ""))
