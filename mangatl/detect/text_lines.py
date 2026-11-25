"""Text-line detection with comic-text-detector (CTD), as used by manga-image-translator,
BallonsTranslator and Koharu. Runs the ONNX export on CPU with onnxruntime.

Post-processing mirrors theirs (DBNet-style): threshold the line probability map at 0.3,
take contours, keep boxes whose mean probability is >= 0.6, and grow each box by
area * 1.5 / perimeter ("unclip") so the glyph edges are inside it.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CTD_URL = ("https://github.com/zyddnys/manga-image-translator/releases/download/"
           "beta-0.3/comictextdetector.pt.onnx")
CTD_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"
INPUT = 1024
BINARY_THRESHOLD = 0.3
SCORE_THRESHOLD = 0.6
UNCLIP_RATIO = 1.5


@dataclass
class TextLine:
    x1: int
    y1: int
    x2: int
    y2: int
    vertical: bool
    score: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def thickness(self) -> int:
        return (self.x2 - self.x1) if self.vertical else (self.y2 - self.y1)


class LineDetector:
    def __init__(self, models_dir: Path):
        import onnxruntime as ort

        path = Path(models_dir) / "detection" / "ctd" / "comictextdetector.pt.onnx"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(CTD_URL, tmp)
            tmp.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != CTD_SHA256:
            raise RuntimeError(f"comic-text-detector checksum mismatch for {path}")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    def detect(self, image: Image.Image) -> list[TextLine]:
        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        scale = INPUT / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        canvas = np.full((INPUT, INPUT, 3), 114, np.uint8)  # YOLO letterbox grey, padded right/bottom
        canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        _, _, det = self.session.run(None, {"images": blob})
        prob = det[0, 0, :nh, :nw]

        bitmap = (prob > BINARY_THRESHOLD).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for c in contours[:1000]:
            if len(c) < 4:
                continue
            (cx, cy), (rw, rh), angle = cv2.minAreaRect(c)
            if min(rw, rh) < 3:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            local = np.zeros((bh, bw), np.uint8)
            cv2.fillPoly(local, [c - [x, y]], 1)
            score = float(prob[y:y + bh, x:x + bw][local > 0].mean()) if local.any() else 0.0
            if score < SCORE_THRESHOLD:
                continue
            d = rw * rh * UNCLIP_RATIO / (2 * (rw + rh))
            box = cv2.boxPoints(((cx, cy), (rw + 2 * d, rh + 2 * d), angle)) / scale
            x1, y1 = np.clip(box.min(0), 0, [w, h]).astype(int)
            x2, y2 = np.clip(box.max(0), 0, [w, h]).astype(int)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(TextLine(int(x1), int(y1), int(x2), int(y2), (y2 - y1) > (x2 - x1), score))
        return lines


def reading_order(lines: list[TextLine], rtl: bool = True) -> list[TextLine]:
    """Order lines in one block: columns right-to-left (vertical) or rows top-to-bottom.

    Lines are clustered into columns/rows first, so a column split in two by a gap is still
    read top-to-bottom before moving to the next column.
    """
    if not lines:
        return []
    vertical = sum(l.vertical for l in lines) * 2 >= len(lines)
    tol = 0.5 * float(np.median([l.thickness for l in lines]))
    across = (lambda l: l.cx) if vertical else (lambda l: l.cy)
    along = (lambda l: l.cy) if vertical else (lambda l: l.cx)
    ordered = sorted(lines, key=across, reverse=vertical and rtl)
    groups: list[list[TextLine]] = []
    for line in ordered:
        if groups and abs(across(line) - np.mean([across(g) for g in groups[-1]])) <= tol:
            groups[-1].append(line)
        else:
            groups.append([line])
    return [l for g in groups for l in sorted(g, key=along)]


_lock = threading.Lock()
_instance: LineDetector | None = None


def get(models_dir: Path) -> LineDetector:
    global _instance
    with _lock:
        if _instance is None:
            _instance = LineDetector(models_dir)
        return _instance
