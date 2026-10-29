"""Build a blind comparison sheet from bake-off logs (eval/bake/out/<label>/<series>.log).

Labels are shuffled into A/B/C...; the mapping is written to key.json (read it only after scoring).
Usage: uv run python eval/bake/sheet.py label1 label2 ... > eval/bake/sheet.md
"""

import json
import random
import re
import sys
from pathlib import Path

BAKE = Path(__file__).resolve().parent
SERIES = ["tojime_no_siora", "balloon_dream", "tencho_isoro"]


def parse(log: Path) -> dict[str, list[tuple[str, str]]]:
    """page -> [(source, translation)] using the final OCR block and the raw LLM response."""
    pages, page, ocr, tr, mode = {}, None, [], [], None
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"Processing \d+/\d+: (\S+)", line)
        if m:
            if page:
                pages[page] = list(zip(ocr, tr))
            page, ocr, tr, mode = m.group(1), [], [], None
            continue
        if line.startswith("Raw OCR output"):
            ocr, mode = [], "ocr"
        elif line.startswith("Raw response"):
            tr, mode = [], "tr"
        elif mode and re.match(r"^\d+: ", line):
            num, text = line.split(": ", 1)
            target = ocr if mode == "ocr" else tr
            if int(num) <= len(target):  # repeated number in the response: the parser keeps one
                continue
            target.append(re.sub(r"^\[(Bubble|OSB)\] ", "", text.strip()))
        elif line.startswith("---") and mode and (ocr if mode == "ocr" else tr):
            mode = None
    if page:
        pages[page] = list(zip(ocr, tr))
    return pages


def main() -> None:
    labels = sys.argv[1:]
    letters = [chr(ord("A") + i) for i in range(len(labels))]
    rng = random.Random()
    shuffled = labels[:]
    rng.shuffle(shuffled)
    key = dict(zip(letters, shuffled))
    (BAKE / "key.json").write_text(json.dumps(key, indent=1))
    ann = {b["book_title"]: b for b in json.load(open(BAKE.parent / "annotation.json"))}

    agree_all = 0
    print("# Blind bake-off sheet\n")
    for s in SERIES:
        runs = {L: parse(BAKE / "out" / key[L] / f"{s}.log") for L in letters}
        for page in sorted(runs[letters[0]]):
            idx = int(Path(page).stem)
            refs = " | ".join(t["text_en"].strip() for t in ann[s]["pages"][idx]["text"])
            print(f"\n## {s} {page}\n\nReference (unaligned): {refs}\n")
            items = runs[letters[0]][page]
            for i, (src, _) in enumerate(items):
                outs = {L: (runs[L].get(page, [])[i][1] if i < len(runs[L].get(page, [])) else "<missing>")
                        for L in letters}
                if len(set(outs.values())) == 1:
                    agree_all += 1
                    print(f"- {i + 1}. `{src}` → all: {outs[letters[0]]}")
                    continue
                print(f"- {i + 1}. `{src}`")
                for L in letters:
                    print(f"    - {L}: {outs[L]}")
    print(f"\n(identical across all models: {agree_all} items)")


if __name__ == "__main__":
    main()
