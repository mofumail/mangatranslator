"""Evaluation aids, switched on by environment variables (eval/harness.py sets them; #7 replaces
the variables with typed settings).

MT_LLM_CACHE=<dir>  LLM responses are stored by request content and replayed when the identical
                    request comes again. llama.cpp isn't bit-exact between runs and one changed
                    line changes the next pages' context, so without this two runs of a
                    typesetting change also differ in a third of their translations.
MT_LLM_FRESH=1      every request is sent with cache_prompt=false: llama.cpp then evaluates the
                    whole prompt anew instead of reusing the previous request's prefix, which
                    makes temperature-0 output repeatable (the reused prefix is what changes the
                    floating-point results). Slower prompt processing. Part of the cache key.
MT_LLM_REPLAY=1     nothing is sent: a request that is not in the cache is appended to
                    MT_LLM_MISSES (a .jsonl file) and fails at once with a non-retryable HTTP
                    error (status 418).
MT_DUMP_TEXT=<dir>  <dir>/<page>.json lists every text region of a page (box, outside-bubble
                    flag, OCR read, translation) in reading order.
"""

from __future__ import annotations

import hashlib
import json
import os
import types
from pathlib import Path
from typing import Any

import requests

_page: dict[str, str | None] = {"name": None, "label": None}


def set_page(image_path: str | Path) -> None:
    """Remember which page is being processed (dump file names, replay-miss log)."""
    p = Path(image_path)
    _page["name"], _page["label"] = p.stem, f"{p.parent.name}/{p.name}"


class _Cached:
    status_code = 200

    def __init__(self, body: Any):
        self._body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self._body


def _strip_images(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_images(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:image"):
        return obj[:40] + "...(image)"
    return obj


def post(url: str, headers: dict | None = None, json: Any = None, timeout: Any = None):  # noqa: A002
    """requests.post for LLM calls, with the MT_LLM_* behaviours above."""
    cache_dir = os.environ.get("MT_LLM_CACHE")
    replay = os.environ.get("MT_LLM_REPLAY") == "1"
    if replay and not cache_dir:
        raise RuntimeError("MT_LLM_REPLAY=1 needs MT_LLM_CACHE")
    if os.environ.get("MT_LLM_FRESH") == "1" and isinstance(json, dict):
        json = {**json, "cache_prompt": False}
    if not cache_dir:
        return requests.post(url, headers=headers, json=json, timeout=timeout)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(_json.dumps([url, json], sort_keys=True).encode()).hexdigest()
    f = Path(cache_dir) / f"{key}.json"
    if f.exists():
        return _Cached(_json.loads(f.read_text()))
    if replay:
        misses = os.environ.get("MT_LLM_MISSES")
        if misses:
            with open(misses, "a", encoding="utf-8") as fh:
                fh.write(_json.dumps({"key": key, "page": _page["label"], "request": _strip_images(json)},
                                     ensure_ascii=False) + "\n")
        miss = types.SimpleNamespace(status_code=418, text=f"replay miss {key[:12]}")
        raise requests.exceptions.HTTPError(f"replay miss {key[:12]}", response=miss)
    response = requests.post(url, headers=headers, json=json, timeout=timeout)
    response.raise_for_status()
    f.write_text(_json.dumps(response.json(), ensure_ascii=False))
    # the request too, minus image data, to see exactly what the model was asked
    f.with_suffix(".req.json").write_text(_json.dumps(_strip_images(json), ensure_ascii=False, indent=1))
    return response


_json = json


def dump_text(bubble_metadata: list[dict], ocr_texts: list[str], texts: list[str]) -> None:
    """MT_DUMP_TEXT: write this page's regions with their OCR read and translation."""
    out_dir = os.environ.get("MT_DUMP_TEXT")
    if not out_dir or not _page["name"]:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    items = [{"bbox": [int(v) for v in (m.get("original_bbox") or m.get("bbox") or [])],
              "osb": bool(m.get("is_outside_text")),
              "ocr": ocr_texts[i] if i < len(ocr_texts) else None,
              "tl": texts[i] if i < len(texts) else None}
             for i, m in enumerate(bubble_metadata)]
    (Path(out_dir) / f"{_page['name']}.json").write_text(_json.dumps(items, ensure_ascii=False, indent=1))
