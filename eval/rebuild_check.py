"""Rebuild every translated page from a project file alone and compare it with the page written.

    python eval/rebuild_check.py FOLDER_OR_PROJECT [...] [--export DIR] [--diff DIR]

Each saved page (not a strip's sections, which are checked through their strip) is rendered
from project.mtl (mangatl.project.rebuild) in this fresh process and compared pixel for pixel
with the PNG the pipeline wrote. Also reports the project's size next to the outputs'.
Exit status 1 if any page differs or fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from mangatl.project.rebuild import rebuild_page  # noqa: E402
from mangatl.project.store import Project  # noqa: E402


def _project_files(arg: str) -> list[Path]:
    """A project file, or every project.mtl under a folder (an output folder, or a harness run)."""
    p = Path(arg)
    found = sorted(p.rglob("project.mtl")) if p.is_dir() else [p] if p.is_file() else []
    if not found:
        sys.exit(f"no project file at or under {p}")
    return found


def check(path: Path, export: Path | None, diff_dir: Path | None) -> tuple[int, int]:
    project = Project(path, create=False)
    ids = project.page_ids()
    saved = [pid for pid in ids if project.load_page(pid).output]
    bad = 0
    out_bytes = 0
    t0 = time.time()
    for pid in saved:
        page = project.load_page(pid)
        out = Path(page.output or "")
        try:
            rebuilt = rebuild_page(project, pid)
            written = Image.open(out)
            written.load()
            out_bytes += out.stat().st_size
            # oxipng stores a page that is really gray (or opaque) in a smaller PNG type, losslessly
            if written.mode != rebuilt.mode and out.suffix.lower() == ".png":
                written = written.convert(rebuilt.mode)
            a, b = np.asarray(rebuilt), np.asarray(written)
            if rebuilt.mode != written.mode or a.shape != b.shape:
                bad += 1
                print(f"  DIFF {pid}: {rebuilt.mode}{rebuilt.size} vs written {written.mode}{written.size}")
            elif not np.array_equal(a, b):
                bad += 1
                changed = (a != b).any(axis=-1) if a.ndim == 3 else a != b
                ys, xs = np.nonzero(changed)
                box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                print(f"  DIFF {pid}: {int(changed.sum())} px differ in {box}")
                if diff_dir:
                    diff_dir.mkdir(parents=True, exist_ok=True)
                    rebuilt.save(diff_dir / f"{pid.replace('/', '_')}.rebuilt.png")
        except Exception as e:  # report and go on
            bad += 1
            print(f"  FAIL {pid}: {e!r}")
        if export:
            target = export / f"{pid.replace('/', '_')}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(project.export_json(pid))
    size = path.stat().st_size
    per = size / max(1, len(saved))
    print(
        f"{path}: {len(saved) - bad}/{len(saved)} pages identical ({len(ids)} documents) in {time.time() - t0:.1f}s; "
        f"project {size / 1e6:.1f} MB ({per / 1e3:.0f} kB/page) vs outputs {out_bytes / 1e6:.1f} MB"
    )
    return len(saved), bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="project files, or folders to search for project.mtl")
    ap.add_argument("--export", type=Path, help="also write each page's document as JSON here")
    ap.add_argument("--diff", type=Path, help="write rebuilt pages that differ here")
    a = ap.parse_args()
    total = bad = 0
    for path in (f for arg in a.paths for f in _project_files(arg)):
        n, b = check(path, a.export, a.diff)
        total, bad = total + n, bad + b
    print(f"TOTAL: {total - bad}/{total} identical")
    sys.exit(1 if bad or not total else 0)


if __name__ == "__main__":
    main()
