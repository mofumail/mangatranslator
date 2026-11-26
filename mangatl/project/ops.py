"""Operations on a page document (as JSON-like data): apply, undo, conflicts, staleness.

Every change is one `Op`: a path into the document, the value before and the value after. Paths
address regions by id ("regions", "r7", "translation"), never by position, so an operation keeps
meaning while other regions come and go. Adding or removing a region is a set of the whole
region to / from None; a pixel edit is a set of the page's `cleaned` patch list (the patches
themselves are blobs, so an operation only stores references).

Undoing an operation restores its old value, but only while the document still holds the value
the operation set; otherwise it raises Conflict instead of silently overwriting a later change.
Changing a region's stage result marks the stages after it stale (detect -> ocr -> translate ->
typeset, clean -> typeset); those status changes are operations in the same group, so undo puts
them back too.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

Path = tuple[str, ...]

# Which stage a region field belongs to, and which stages depend on each stage
FIELD_STAGE = {
    "bbox": "detect",
    "seg_mask": "detect",
    "kind": "detect",
    "neighbors": "detect",
    "ocr": "ocr",
    "skip": "ocr",
    "translation": "translate",
    "cleaning": "clean",
    "pending_fill": "clean",
    "outside": "typeset",
}
DOWNSTREAM = {
    "detect": ("ocr", "translate", "clean", "typeset"),
    "ocr": ("translate", "typeset"),
    "translate": ("typeset",),
    "clean": ("typeset",),
    "typeset": (),
}


class Conflict(Exception):
    """The document no longer holds what an operation expects."""


@dataclass(frozen=True)
class Op:
    path: Path
    old: Any
    new: Any

    def inverse(self) -> Op:
        return Op(self.path, self.new, self.old)

    def to_json(self) -> dict:
        return {"path": list(self.path), "old": self.old, "new": self.new}

    @classmethod
    def from_json(cls, data: dict) -> Op:
        return cls(tuple(data["path"]), data["old"], data["new"])


def _region_index(doc: dict, region_id: str) -> int | None:
    for i, r in enumerate(doc.get("regions", [])):
        if r.get("id") == region_id:
            return i
    return None


def get(doc: dict, path: Path) -> Any:
    """The value at `path`, or None when it doesn't exist."""
    node: Any = doc
    i = 0
    while i < len(path):
        key = path[i]
        if key == "regions" and isinstance(node, dict) and i + 1 < len(path):
            idx = _region_index(node, path[i + 1])
            if idx is None:
                return None
            node = node["regions"][idx]
            i += 2
            continue
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
        i += 1
    return node


def put(doc: dict, path: Path, value: Any) -> None:
    """Set `path` to `value`. For a whole region (("regions", id)) None removes it and a dict
    adds or replaces it; for a key of a mapping (e.g. a stage status) None removes the key."""
    if len(path) == 2 and path[0] == "regions":
        idx = _region_index(doc, path[1])
        if value is None:
            if idx is not None:
                del doc["regions"][idx]
        elif idx is None:
            doc.setdefault("regions", []).append(copy.deepcopy(value))
        else:
            doc["regions"][idx] = copy.deepcopy(value)
        return
    parent_path, key = path[:-1], path[-1]
    parent = get(doc, parent_path) if parent_path else doc
    if not isinstance(parent, dict):
        raise Conflict(f"no {'/'.join(parent_path)} to set {key} in")
    if value is None and key in parent and parent_path and parent_path[-1] == "status":
        del parent[key]
    else:
        parent[key] = copy.deepcopy(value)


def set_op(doc: dict, path: Path, value: Any) -> list[Op]:
    """The operations that set `path` to `value`: the change itself, plus marking later stages
    stale when it changes a region's stage result. Empty when nothing would change."""
    old = get(doc, path)
    if old == value:
        return []
    ops = [Op(path, copy.deepcopy(old), copy.deepcopy(value))]
    if len(path) >= 3 and path[0] == "regions" and path[2] in FIELD_STAGE:
        region_id, stage = path[1], FIELD_STAGE[path[2]]
        status = get(doc, ("regions", region_id, "status")) or {}
        for later in DOWNSTREAM[stage]:
            if status.get(later) in ("done", "user"):
                ops.append(Op(("regions", region_id, "status", later), status[later], "stale"))
        if status.get(stage) != "user":
            ops.append(Op(("regions", region_id, "status", stage), status.get(stage), "user"))
    return ops


def apply(doc: dict, ops: Iterable[Op], check: bool = False) -> None:
    """Apply operations in order. With `check`, each must find its old value (accepting a
    proposal made against an earlier state)."""
    ops = list(ops)
    if check:
        scratch = copy.deepcopy(doc)
        for op in ops:
            if get(scratch, op.path) != op.old:
                raise Conflict(f"{'/'.join(op.path)} changed since the operation was made")
            put(scratch, op.path, op.new)
    for op in ops:
        put(doc, op.path, op.new)


def undo(doc: dict, ops: Iterable[Op]) -> None:
    """Undo operations (one, or a whole group) atomically, newest first."""
    ops = list(ops)
    scratch = copy.deepcopy(doc)
    for op in reversed(ops):
        if get(scratch, op.path) != op.new:
            raise Conflict(f"{'/'.join(op.path)} was changed after this operation")
        put(scratch, op.path, op.old)
    for op in reversed(ops):
        put(doc, op.path, op.old)
