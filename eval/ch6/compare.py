"""Compare a run with the professional scanlation.

  compare.py <label> [other_label] [--pages 003,004] [--sheet]

Prints, per region: Japanese OCR, our translation (and other_label's, matched by box), and the
pro English read by refs.py. Ends with a chrF score against the pro text (character n-gram F2,
a rough similarity signal; the pro translation is one good answer, not the only one).
--sheet also writes out/<label>/sheet/<page>.jpg: our page next to the pro page.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
CACHE = HERE / "ref_cache.json"


def norm(t: str | None) -> str:
    t = re.sub(r"[*_]+", "", t or "")
    t = t.replace("…", "...").replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(t.lower().split())


def chrf(hyp: str, ref: str, n_max: int = 6, beta: float = 2.0) -> float:
    hyp, ref = hyp.replace(" ", ""), ref.replace(" ", "")
    precs, recs = [], []
    for n in range(1, n_max + 1):
        h = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        r = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / sum(h.values()))
        recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    return 0.0 if p + r == 0 else (1 + beta**2) * p * r / (beta**2 * p + r)


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


def load(label: str) -> dict[str, list]:
    d = HERE / "out" / label / "text"
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pages = None
    if "--pages" in sys.argv:
        pages = set(sys.argv[sys.argv.index("--pages") + 1].split(","))
        args = [a for a in args if a != sys.argv[sys.argv.index("--pages") + 1]]
    label, other = args[0], (args[1] if len(args) > 1 else None)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    run, run2 = load(label), (load(other) if other else {})
    scores, scores2 = [], []
    for page, items in run.items():
        if pages and page not in pages:
            continue
        print(f"\n=== {page}")
        for i, it in enumerate(items, 1):
            ref = cache.get(f"{page}:{','.join(map(str, it['bbox']))}")
            o = None
            if other:
                cand = [(iou(it["bbox"], j["bbox"]), j) for j in run2.get(page, []) if j["bbox"]]
                best = max(cand, key=lambda c: c[0], default=(0, None))
                o = best[1] if best[0] >= 0.5 else None
            kind = "OSB" if it["osb"] else "BUB"
            print(f"{i:>2} {kind} JP : {(it['ocr'] or '').strip()}")
            print(f"       {label:<5}: {it['tl']}")
            if other:
                print(f"       {other:<5}: {o['tl'] if o else '(no matching box)'}")
            print(f"       pro  : {ref if ref is not None else '(not read)'}")
            if ref:
                scores.append(chrf(norm(it["tl"]), norm(ref)))
                if o:
                    scores2.append(chrf(norm(o["tl"]), norm(ref)))
        if "--sheet" in sys.argv:
            ours = HERE / "out" / label / f"{page}_translated.png"
            if not ours.exists():
                ours = next((HERE / "out" / label).glob(f"{page}_translated.*"), None)
            if ours:
                a, b = Image.open(ours).convert("RGB"), Image.open(HERE / "en" / f"{page}.jpg").convert("RGB")
                h = 1400
                a = a.resize((round(a.width * h / a.height), h))
                b = b.resize((round(b.width * h / b.height), h))
                s = Image.new("RGB", (a.width + b.width + 20, h), "white")
                s.paste(a, (0, 0)); s.paste(b, (a.width + 20, 0))
                (HERE / "out" / label / "sheet").mkdir(exist_ok=True)
                s.save(HERE / "out" / label / "sheet" / f"{page}.jpg", quality=88)
    if scores:
        print(f"\nchrF vs pro: {label} {100 * sum(scores) / len(scores):.1f} over {len(scores)} regions")
    if scores2:
        print(f"chrF vs pro: {other} {100 * sum(scores2) / len(scores2):.1f} over {len(scores2)} matched regions")


if __name__ == "__main__":
    main()
