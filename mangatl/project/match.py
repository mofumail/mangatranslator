"""Matching a page's regions to a new detection, so region ids (and the work attached to them)
survive re-detecting a page or rescanning part of it.

Pairs are made greedily by overlap (intersection over union), best first, between regions of
the same kind. A matched detection takes the region's id; an unmatched one is new; a region
nothing matched has vanished (the caller keeps it, flagged, when it carries a person's work).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

Box = tuple[int, int, int, int]
MIN_IOU = 0.3  # below this two boxes are different text
SAME_IOU = 0.9  # at or above this a match is the same region, its reads still valid


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlaps(box: Box, area: Box) -> bool:
    """Whether a region lies mostly inside `area` (at least half of it)."""
    ix = max(0, min(box[2], area[2]) - max(box[0], area[0]))
    iy = max(0, min(box[3], area[3]) - max(box[1], area[1]))
    own = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    return ix * iy * 2 >= own


class Matching(NamedTuple):
    matched: dict[int, tuple[str, float]]  # new index -> (old id, overlap)
    added: list[int]  # new indices nothing matched
    vanished: list[str]  # old ids nothing matched


def match(
    old: Sequence[tuple[str, str, Box]], new: Sequence[tuple[str, Box]], min_iou: float = MIN_IOU
) -> Matching:
    """old: (id, kind, box) per existing region; new: (kind, box) per detection."""
    candidates = sorted(
        (
            (iou(ob, nb), oi, ni)
            for oi, (_, okind, ob) in enumerate(old)
            for ni, (nkind, nb) in enumerate(new)
            if okind == nkind
        ),
        reverse=True,
    )
    used_old: set[int] = set()
    matched: dict[int, tuple[str, float]] = {}
    for score, oi, ni in candidates:
        if score < min_iou:
            break
        if oi in used_old or ni in matched:
            continue
        used_old.add(oi)
        matched[ni] = (old[oi][0], score)
    return Matching(
        matched,
        [ni for ni in range(len(new)) if ni not in matched],
        [old[oi][0] for oi in range(len(old)) if oi not in used_old],
    )
