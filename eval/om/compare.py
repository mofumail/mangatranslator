"""Compare OpenMantra runs with the book's professional English (eval/openmantra/annotation.json).

  compare.py <book> <label> [label ...] [--diff]

Every annotated text box is matched to the region of each run that covers most of it, and the
Japanese, each run's translation and the pro English are printed. --diff prints only boxes where
the runs disagree. Ends with chrF against the pro text (a rough signal, not a verdict).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch6"))
from compare import chrf, norm  # noqa: E402

HERE = Path(__file__).resolve().parent
ANN = HERE.parent / "openmantra" / "annotation.json"


def cover(a, b) -> float:
    """Share of box a (x0, y0, x1, y1) covered by box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area else 0.0


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    book, labels = args[0], args[1:]
    only_diff = "--diff" in sys.argv
    pages = next(b for b in json.loads(ANN.read_text()) if b["book_title"] == book)["pages"]
    runs = {lb: HERE / "out" / lb / book / "text" for lb in labels}
    scores = {lb: [] for lb in labels}
    width = max(4, *(len(lb) for lb in labels))
    for page in pages:
        stem = Path(page["image_paths"]["ja"]).stem
        items = {}
        for lb, d in runs.items():
            f = d / f"{stem}.json"
            items[lb] = json.loads(f.read_text()) if f.exists() else []
        header = False
        for t in page["text"]:
            box = (t["x"], t["y"], t["x"] + t["w"], t["y"] + t["h"])
            got = {}
            for lb in labels:
                cand = [(cover(box, it["bbox"]), it) for it in items[lb] if len(it["bbox"]) == 4]
                c, it = max(cand, key=lambda c: c[0], default=(0, None))
                got[lb] = it["tl"] if it and c >= 0.3 else None
            if only_diff and len(set(got.values())) <= 1:
                continue
            if not header:
                print(f"\n=== {stem}")
                header = True
            print(f"  JP  : {t['text_ja']}")
            for lb in labels:
                print(f"  {lb:<{width}}: {got[lb] if got[lb] is not None else '(missed)'}")
                if got[lb]:
                    scores[lb].append(chrf(norm(got[lb]), norm(t["text_en"])))
            print(f"  {'pro':<{width}}: {t['text_en']}")
    for lb in labels:
        s = scores[lb]
        if s:
            print(f"chrF vs pro: {lb} {100 * sum(s) / len(s):.1f} over {len(s)} boxes")


if __name__ == "__main__":
    main()
