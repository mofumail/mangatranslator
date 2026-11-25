"""Regression harness: run the eval sets, then show exactly what changed against a saved baseline.

    uv run python eval/harness.py baseline              # live LLM; then replays itself as a self-check
    uv run python eval/harness.py check                 # replay against eval/runs/baseline, no LLM server
    uv run python eval/harness.py check --live          # cache misses go to the LLM instead of failing
    uv run python eval/harness.py compare RUN_A RUN_B   # diff two existing runs

A baseline stores the output pages, a text dump per page (OCR + translation per region) and every LLM
response (--llm-cache). `check` re-runs the pipeline with those responses replayed (--llm-replay), so
detection, OCR, cleaning and typesetting changes show up as image/text diffs within minutes, and any change
to what the translator is asked shows up as a cache miss (the prompt changed). cache_prompt=false is set on
every run (--llm-fresh): it makes temperature-0 answers repeatable, and it is part of the cache key.

Runs go to eval/runs/<name>/ (git-ignored). Sets whose pages are not on this machine are skipped.
Exit code of check/compare: 0 no differences, 1 differences, 2 error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "eval" / "runs"

# Fixed for every run: the port is part of the LLM cache key, PNG avoids JPEG noise in pixel diffs.
COMMON = [
    "--port",
    "8090",
    "--set",
    "translation.temperature=0",
    "--format",
    "png",
    "--verbose",
    "--llm-fresh",
]


@dataclass
class Unit:
    """One translate.py invocation (one chapter: its own glossary, context and text dump)."""

    name: str
    input: str
    args: list[str] = field(default_factory=list)


SETS: dict[str, list[Unit]] = {
    # 10 OpenMantra pages tracked in git (CC BY-NC 4.0, eval only)
    "om10": [Unit(f"om10/{c}", f"eval/in/{c}") for c in ("tojime_no_siora", "balloon_dream", "tencho_isoro")],
    # shounen chapter with an English scanlation (local only)
    "ch6": [Unit("ch6", "eval/ch6/jp")],
    # doujin CG pages from 8 works (local only)
    "djn": [Unit("djn", "eval/djn/in", ["--adult"])],
}
DEFAULT_SETS = ["om10", "ch6", "djn"]

PIXEL_TOL = 0  # a pixel counts as changed when any channel differs by more than this
MIN_CHANGED = 0  # pages with at most this many changed pixels count as unchanged


def log(msg: str) -> None:
    print(f"[harness] {msg}", flush=True)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def model_listing() -> list[list]:
    """Model files (> 1 MB) with sizes, so 'everything changed' can be traced to a model update."""
    out = []
    for base in (ROOT / "models",):
        for f in sorted(base.rglob("*")):
            try:
                if f.is_file() and f.stat().st_size > 1 << 20:
                    out.append([str(f.relative_to(ROOT)), f.stat().st_size])
            except OSError:
                pass
    return out


def units_for(sets: list[str]) -> list[Unit]:
    units = []
    for s in sets:
        if s not in SETS:
            sys.exit(f"unknown set {s!r}; known: {', '.join(SETS)}")
        for u in SETS[s]:
            src = ROOT / u.input
            if src.is_dir() and any(src.iterdir()):
                units.append(u)
            else:
                log(f"skip {u.name}: {u.input} not on this machine")
    return units


# ---------------------------------------------------------------- running


def run_unit(unit: Unit, run_dir: Path, cache: Path, replay: bool, extra: list[str] | None = None) -> dict:
    d = run_dir / unit.name
    d.mkdir(parents=True)
    out = d / "out"
    argv = [
        sys.executable,
        str(ROOT / "translate.py"),
        "--input",
        str(ROOT / unit.input),
        "--output",
        str(out),
        *COMMON,
        "--llm-cache",
        str(cache),
        "--dump-text",
        str(d / "text"),
        *unit.args,
        *(extra or []),
    ]
    if replay:
        argv += ["--llm-replay", "--llm-misses", str(d / "misses.jsonl")]
    env = dict(os.environ)
    t0 = time.time()
    with open(d / "log.txt", "w") as fh:
        rc = subprocess.call(argv, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    secs = time.time() - t0
    pages = len(list(out.glob("*.png"))) if out.exists() else 0
    log(f"{unit.name}: rc={rc} pages={pages} {secs:.0f}s")
    return {
        "unit": unit.name,
        "argv": [str(a) for a in argv[1:]],
        "extra": list(extra or []),
        "rc": rc,
        "seconds": round(secs, 1),
        "pages": pages,
        "replay": replay,
        "cache": str(cache.relative_to(ROOT)),
    }


def new_run_dir(name: str, force: bool) -> Path:
    d = RUNS / name
    if d.exists():
        if not force:
            sys.exit(f"{d.relative_to(ROOT)} exists; pick another --name or pass --force to replace it")
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def write_meta(run_dir: Path, kind: str, results: list[dict], against: str | None = None) -> None:
    meta = {
        "kind": kind,
        "against": against,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_rev": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "units": results,
        "models": model_listing(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def cmd_baseline(a) -> int:
    run_dir = new_run_dir(a.name, a.force)
    units = units_for(a.sets)
    extra = shlex.split(a.extra_args) if a.extra_args else []
    results = [run_unit(u, run_dir, run_dir / u.name / "llm", replay=False, extra=extra) for u in units]
    write_meta(run_dir, "baseline", results)
    if any(r["rc"] for r in results):
        log("a unit failed; see its log.txt")
        return 2
    if a.no_self_check:
        return 0
    log("self-check: replaying the baseline (expect no differences)")
    a2 = argparse.Namespace(
        against=a.name,
        name=f"{a.name}-selfcheck",
        force=True,
        live=False,
        tol=a.tol,
        only=None,
        extra_args=None,
    )
    return cmd_check(a2)


def cmd_check(a) -> int:
    base = RUNS / a.against
    meta = json.loads((base / "meta.json").read_text())
    run_dir = new_run_dir(a.name or time.strftime("check-%Y%m%d-%H%M%S"), a.force)
    known = {u.name: u for s in SETS.values() for u in s}
    results = []
    only = a.only.split(",") if a.only else None
    extra = shlex.split(a.extra_args) if a.extra_args else []
    for r in meta["units"]:
        u = known.get(r["unit"])
        if u is None:
            log(f"skip {r['unit']}: no longer defined in SETS")
            continue
        if only and not any(r["unit"] == o or r["unit"].startswith(o + "/") for o in only):
            continue
        cache = base / u.name / "llm"
        if a.live:  # misses are sent to the LLM and stored in this run's own copy of the cache
            own = run_dir / "_llm" / u.name
            shutil.copytree(cache, own, copy_function=os.link)
            cache = own
        # a variant baseline (baseline --extra-args) is replayed with the same flags
        results.append(run_unit(u, run_dir, cache, replay=not a.live, extra=r.get("extra", []) + extra))
    write_meta(run_dir, "check", results, against=a.against)
    return report(base, run_dir, a.tol)


def cmd_compare(a) -> int:
    return report(RUNS / a.a, RUNS / a.b, a.tol)


# ---------------------------------------------------------------- comparing


def load_dump(d: Path) -> dict[str, list]:
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))} if d.exists() else {}


def iou(a, b) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def text_changes(old: list, new: list) -> list[str]:
    """Region-by-region differences in OCR/translation. Same count: by index; otherwise by box overlap."""
    lines = []
    if len(old) == len(new) and all(iou(o["bbox"], n["bbox"]) > 0.5 for o, n in zip(old, new)):
        pairs = list(zip(range(len(old)), old, new))
    else:
        lines.append(f"regions: {len(old)} -> {len(new)}")
        used, pairs = set(), []
        for i, o in enumerate(old):
            best = max(
                ((iou(o["bbox"], n["bbox"]), j) for j, n in enumerate(new) if j not in used), default=(0, -1)
            )
            if best[0] > 0.3:
                used.add(best[1])
                pairs.append((i, o, new[best[1]]))
            else:
                lines.append(f"  - region {i + 1} gone: {o.get('ocr')!r} -> {o.get('tl')!r}")
        for j, n in enumerate(new):
            if j not in used:
                lines.append(f"  + new region at {n['bbox']}: {n.get('ocr')!r} -> {n.get('tl')!r}")
    for i, o, n in pairs:
        for k, label in (("ocr", "OCR"), ("tl", "TL")):
            if o.get(k) != n.get(k):
                lines.append(
                    f"  region {i + 1} {label}: {o.get(k)!r}\n  {' ' * len(f'region {i + 1} {label}:')} -> {n.get(k)!r}"
                )
    return lines


def image_change(a: Path, b: Path, tol: int):
    ia, ib = np.asarray(Image.open(a).convert("RGB")), np.asarray(Image.open(b).convert("RGB"))
    if ia.shape != ib.shape:
        return None, ia, ib
    mask = np.abs(ia.astype(np.int16) - ib.astype(np.int16)).max(axis=2) > tol
    return mask, ia, ib


def diff_png(ia, ib, mask, path: Path) -> None:
    """baseline | new | new dimmed with changed pixels in red, scaled to at most 1600 px high."""
    hl = (ib * 0.35 + 165).astype(np.uint8)
    hl[mask] = (230, 0, 0)
    gap = np.full((ia.shape[0], 12, 3), 255, np.uint8)
    im = Image.fromarray(np.concatenate([ia, gap, ib, gap, hl], axis=1))
    if im.height > 1600:
        im = im.resize((round(im.width * 1600 / im.height), 1600), Image.LANCZOS)
    im.save(path, quality=88)


def changed_regions(mask, regions: list) -> list[str]:
    out = []
    for i, r in enumerate(regions):
        if len(r.get("bbox", [])) != 4:
            continue
        x0, y0, x1, y1 = (max(0, v) for v in r["bbox"])
        sub = mask[y0:y1, x0:x1]
        if sub.size and sub.any():
            out.append(f"region {i + 1} ({100 * sub.mean():.1f}% of box): {r.get('tl')!r}")
    return out


def misses_of(d: Path) -> list[dict]:
    """Unique cache misses in order (upstream retries a failed page once, so each appears twice)."""
    f = d / "misses.jsonl"
    seen, out = set(), []
    for line in f.read_text().splitlines() if f.exists() else []:
        m = json.loads(line)
        if m["key"] not in seen:
            seen.add(m["key"])
            out.append(m)
    return out


def request_lines(req: dict) -> list[str]:
    lines = []
    for msg in req.get("messages", []):
        c = msg.get("content")
        parts = [c] if isinstance(c, str) else [p.get("text") or f"<{p.get('type')}>" for p in c or []]
        for part in parts:
            lines += [f"[{msg.get('role')}] {line}" for line in str(part).splitlines()]
    return lines


def request_diff(miss: dict, cache: Path, limit: int = 12) -> list[str]:
    """What changed in a missed request: diff against the most similar request in the baseline cache."""
    import difflib

    new = request_lines(miss["request"])
    best, best_lines = 0.0, None
    for f in cache.glob("*.req.json"):
        old = request_lines(json.loads(f.read_text()))
        r = difflib.SequenceMatcher(None, old, new, autojunk=False).ratio()
        if r > best:
            best, best_lines = r, old
    if best_lines is None:
        return ["(no baseline requests to compare with)"]
    out = [
        ln
        for ln in difflib.unified_diff(best_lines, new, lineterm="", n=0)
        if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
    ]

    def clip(i: int, ln: str) -> str:  # long prompt lines: show the part around the change
        if len(ln) < 220:
            return ln
        other = next((o for o in out[max(0, i - 1) : i + 2] if o[:1] != ln[:1]), "")
        k = next((j for j, (x, y) in enumerate(zip(ln[1:], other[1:])) if x != y), 0)
        start = max(0, k - 80)
        return ln[0] + ("..." if start else "") + ln[1 + start : 1 + start + 200] + "..."

    shown = [clip(i, ln) for i, ln in enumerate(out[:limit])]
    if len(out) > limit:
        shown.append(f"... {len(out) - limit} more changed lines")
    return shown or ["(identical text; options such as temperature or max_tokens differ)"]


_ERROR_LINE = re.compile(r"error|exception|traceback|failed", re.I)
_EXPECTED = re.compile(
    r"replay miss|All bubbles failed|Retry failed|still failed|failed image|Finished in", re.I
)


def new_errors(base_log: Path, run_log: Path, limit: int = 8) -> list[str]:
    """Error-looking log lines of this run that the baseline's log doesn't have (numbers and
    paths normalized). The pipeline catches many errors and carries on, so a broken stage can
    look like missing regions rather than a crash."""

    def lines(f: Path) -> dict[str, str]:
        out = {}
        for line in f.read_text(errors="replace").splitlines() if f.exists() else []:
            if _ERROR_LINE.search(line) and not _EXPECTED.search(line):
                key = re.sub(r"\d+(\.\d+)?", "#", re.sub(r"/\S+", "<path>", line.strip()))
                out.setdefault(key, line.strip())
        return out

    base, run = lines(base_log), lines(run_log)
    fresh = [run[k] for k in run if k not in base]
    return [ln[:200] for ln in fresh[:limit]] + (
        [f"... {len(fresh) - limit} more"] if len(fresh) > limit else []
    )


def report(base: Path, run: Path, tol: int) -> int:
    diffs = run / "diffs"
    shutil.rmtree(diffs, ignore_errors=True)
    rows, details, total = [], [], 0
    bmeta = json.loads((base / "meta.json").read_text())
    rmeta = json.loads((run / "meta.json").read_text())
    rtimes = {u["unit"]: u for u in rmeta["units"]}
    for bu in bmeta["units"]:
        name = bu["unit"]
        bd, rd = base / name, run / name
        if not rd.exists():
            continue
        bout, rout = bd / "out", rd / "out"
        bpages = {p.name: p for p in bout.glob("*.png")} if bout.exists() else {}
        rpages = {p.name: p for p in rout.glob("*.png")} if rout.exists() else {}
        bdump, rdump = load_dump(bd / "text"), load_dump(rd / "text")
        misses = misses_of(rd)
        img_changed = txt_changed = 0
        unit_lines = []
        if rtimes.get(name, {}).get("rc"):
            unit_lines.append(
                f"**translate.py exited with code {rtimes[name]['rc']}** (see `{name}/log.txt`)"
            )
        missed = {Path(m.get("page") or "?").stem for m in misses}
        errs = new_errors(bd / "log.txt", rd / "log.txt")
        if errs:
            unit_lines.append(
                "**New error lines in the log** (not in the baseline's):\n```\n" + "\n".join(errs) + "\n```"
            )
        if misses:
            first = misses[0]
            unit_lines.append(
                f"**{len(misses)} LLM cache misses** on {len(missed)} page(s), first on "
                f"`{first.get('page')}` (later misses in a chapter usually follow from the first: "
                f"context and glossary carry it forward). What changed in the first request, "
                f"against the closest baseline request:\n```diff\n"
                + "\n".join(request_diff(first, bd / "llm"))
                + "\n```"
            )
        for page in sorted(set(bpages) | set(rpages)):
            stem = page.removesuffix(".png").removesuffix("_translated")
            lines = []
            if page not in rpages:
                lines.append(
                    "not rendered: its translation was a cache miss"
                    if stem in missed
                    else "page missing in this run"
                )
            elif page not in bpages:
                lines.append("page not in baseline")
            else:
                mask, ia, ib = image_change(bpages[page], rpages[page], tol)
                if mask is None:
                    lines.append(
                        f"image size changed {ia.shape[1]}x{ia.shape[0]} -> {ib.shape[1]}x{ib.shape[0]}"
                    )
                    img_changed += 1
                elif int(mask.sum()) > MIN_CHANGED:
                    img_changed += 1
                    diffs.mkdir(exist_ok=True)
                    png = diffs / f"{name.replace('/', '__')}__{stem}.jpg"
                    diff_png(ia, ib, mask, png)
                    lines.append(
                        f"image: {int(mask.sum())} px changed ({100 * mask.mean():.2f}%) "
                        f"[side by side]({png.relative_to(run)})"
                    )
                    lines += ["  " + s for s in changed_regions(mask, bdump.get(stem, []))]
            if stem in bdump and stem not in rdump and page in rpages:
                lines.append("no text dump in this run (translation call failed on this page?)")
            elif stem in bdump and stem in rdump:
                t = text_changes(bdump[stem], rdump[stem])
                if t:
                    txt_changed += 1
                    lines += t
            if lines:
                unit_lines.append(f"- `{stem}`\n" + "\n".join(f"  {s}" for s in lines))
        bglos, rglos = bout / "glossary.txt", rout / "glossary.txt"
        if (bglos.read_text() if bglos.exists() else "") != (rglos.read_text() if rglos.exists() else ""):
            unit_lines.append("- glossary.txt differs")
        n = len(unit_lines)
        total += n
        rows.append(
            f"| {name} | {len(bpages)} | {img_changed} | {txt_changed} | {len(misses)} | "
            f"{bu.get('seconds', '?')}s / {rtimes.get(name, {}).get('seconds', '?')}s |"
        )
        if unit_lines:
            details.append(f"\n## {name}\n\n" + "\n".join(unit_lines))
    head = (
        f"# {run.name} vs {base.name}\n\n"
        f"baseline {bmeta['git_rev'][:10]}{' (dirty)' if bmeta['git_dirty'] else ''}, "
        f"this run {rmeta['git_rev'][:10]}{' (dirty)' if rmeta['git_dirty'] else ''}; "
        f"pixel tolerance {tol}\n"
    )
    if bmeta.get("models") != rmeta.get("models"):
        head += "\n**Model files differ between the two runs** (meta.json `models`).\n"
    table = (
        "\n| unit | pages | image changed | text changed | cache misses | time base / run |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    verdict = "\nNo differences.\n" if not total else ""
    text = head + table + verdict + "".join(details) + "\n"
    (run / "report.md").write_text(text)
    print(text)
    log(f"report: {(run / 'report.md').relative_to(ROOT)}")
    return 1 if total else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline", help="run the sets with the live LLM and store everything")
    b.add_argument("--name", default="baseline")
    b.add_argument(
        "--sets",
        type=lambda s: s.split(","),
        default=DEFAULT_SETS,
        help=f"comma-separated, from: {', '.join(SETS)}",
    )
    b.add_argument("--force", action="store_true", help="replace an existing run of that name")
    b.add_argument("--no-self-check", action="store_true")
    b.add_argument(
        "--extra-args",
        default=None,
        help='extra translate.py flags for a variant baseline, e.g. "--vision full"; check '
        "replays it with the same flags",
    )
    c = sub.add_parser("check", help="re-run with the baseline's LLM answers and report differences")
    c.add_argument("--against", default="baseline")
    c.add_argument("--name", default=None, help="default: check-<timestamp>")
    c.add_argument("--force", action="store_true")
    c.add_argument("--live", action="store_true", help="send cache misses to the LLM instead of failing them")
    c.add_argument("--only", default=None, help="comma-separated units or sets to run, e.g. ch6,om10")
    c.add_argument(
        "--extra-args",
        default=None,
        help='extra translate.py flags for a variant run, e.g. "--sfx redraw" (render-only '
        "variants replay without misses; compare a pre/post pair of such runs)",
    )
    m = sub.add_parser("compare", help="report differences between two existing runs")
    m.add_argument("a")
    m.add_argument("b")
    for q in (b, c, m):
        q.add_argument("--tol", type=int, default=PIXEL_TOL, help="per-channel pixel tolerance")
    a = p.parse_args()
    sys.exit({"baseline": cmd_baseline, "check": cmd_check, "compare": cmd_compare}[a.cmd](a))


if __name__ == "__main__":
    main()
