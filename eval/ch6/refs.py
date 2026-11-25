"""Professional-translation reference for each text region of a run.

  refs.py <label> [--url http://127.0.0.1:8091]

For every page, the English scanlation page (eval/ch6/en) is aligned to the Japanese page
(feature matching, similarity transform), each region box from the run's text dump
(out/<label>/text/<page>.json) is mapped onto it, and a vision LLM transcribes the lettering
in that crop. Reads are cached by page and box in eval/ch6/ref_cache.json, so later runs only
read new boxes. Needs a llama-server with a vision projector at --url.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CACHE = HERE / "ref_cache.json"
PROMPT = (
    "This is a crop of an English manga page. Transcribe the English lettering that sits in "
    "the middle of the crop (a speech bubble, caption or sound effect), exactly as written, "
    "joining its lines with spaces. Ignore text cut off at the crop edges. If there is no "
    "English text, answer with -. Output only the transcription."
)


def align(jp: np.ndarray, en: np.ndarray) -> np.ndarray:
    """2x3 matrix mapping Japanese page pixels to English page pixels."""
    orb = cv2.ORB_create(5000)
    g1 = cv2.cvtColor(jp, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(en, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    sx, sy = en.shape[1] / jp.shape[1], en.shape[0] / jp.shape[0]
    fallback = np.array([[sx, 0, 0], [0, sy, 0]], np.float32)
    if d1 is None or d2 is None:
        return fallback
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 20:
        return fallback
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    m, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=4)
    if m is None or inliers.sum() < 20 or abs(m[0, 0] - sx) > 0.15:
        return fallback
    return m


def read(url: str, crop: Image.Image) -> str:
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 200,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["choices"][0]["message"]["content"].strip()
    return "" if text in ("-", "") else " ".join(text.split())


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    url = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "http://127.0.0.1:8091"
    label = args[0]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    dump = HERE / "out" / label / "text"
    for f in sorted(dump.glob("*.json")):
        page = f.stem
        items = json.loads(f.read_text())
        jp = cv2.imread(str(HERE / "jp" / f"{page}.jpg"))
        en_path = HERE / "en" / f"{page}.jpg"
        if jp is None or not en_path.exists():
            continue
        en = cv2.imread(str(en_path))
        m = align(jp, en)
        en_pil = Image.open(en_path)
        W, H = en_pil.size
        for it in items:
            if len(it["bbox"]) != 4:
                continue
            key = f"{page}:{','.join(map(str, it['bbox']))}"
            if key in cache:
                continue
            x0, y0, x1, y1 = it["bbox"]
            pts = np.float32([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
            q = pts @ m[:, :2].T + m[:, 2]
            ex0, ey0 = q.min(axis=0)
            ex1, ey1 = q.max(axis=0)
            # English lettering often spills past the Japanese text area
            px, py = 0.25 * (ex1 - ex0) + 8, 0.15 * (ey1 - ey0) + 8
            box = (max(0, int(ex0 - px)), max(0, int(ey0 - py)), min(W, int(ex1 + px)), min(H, int(ey1 + py)))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                cache[key] = ""
                continue
            cache[key] = read(url, en_pil.crop(box))
            print(f"{key}: {cache[key]}", flush=True)
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
