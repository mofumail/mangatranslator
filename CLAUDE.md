# mangatranslator

A manga translation studio driven by agents. `studio.py` runs a local server (UI on 7870, MCP at
`/mcp`): it drafts pages (detect, clean, OCR, size the lettering), and an agent (e.g. a separate
Claude Code instance with the MCP added) translates and letters them. The agent-side workflow is
`mangatl/studio/playbook.md`. `translate.py` / `webui.py` are the old batch pipeline with a local LLM,
being removed (#32).

## How we work
- Incremental, per run: the user runs the studio on a chapter with a separate Claude instance (kept
  apart so its judgment stays impartial), brings back its feedback, we file it as an issue and
  implement it. The old roadmap issues are ignored.
- Zero backwards compatibility: replace outdated systems outright; no shims, no legacy paths.
- Commit and push only when the user says so. End commit messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Leave the user's processes alone: the web UI on 7860 and their studio on 7870. Experiments use
  7871 (studio) and 8090 / 8091 (llama-server). No `pkill -f`.
- Never send the user's email or HF token anywhere. Adult (R18) pages are fine to process.

## Checks before every commit
```bash
uv run ruff check . && uv run ruff format --check .
uv run --group dev pyright
uv run python -m pytest -q --ignore=tests/test_engine_gpu.py --ignore=tests/test_stages_gpu.py --ignore=tests/test_studio_gpu.py
uv run python -m pytest -q tests/test_studio_gpu.py tests/test_stages_gpu.py   # GPU, ~1 min
uv run python eval/harness.py check --name X --force --only om10   # then ch6,djn for renderer/cleaning changes
uv run python eval/rebuild_check.py eval/runs/X                    # every page rebuilt from project.mtl
```
- Always `uv run python -m pytest`; plain `uv run pytest` fails with ModuleNotFoundError.
- The harness must report zero differences unless a change is meant to alter output. Variant
  checks (sfx redraw, keep-other-language, vision full/off, a small VRAM budget) live in the
  scratchpad script `allcheck11.sh`; recreate it from `eval/README.md` if it's gone.
- CI runs ruff, pyright and the light tests with a minimal package set (see
  `.github/workflows/ci.yml`): light tests must not import torch or cv2.

## Gotchas
- `mangatl/project/capture.py` `READ_KEYS`: every region key typesetting reads must be listed there
  (and carried by capture `_region` / `_item`); `tests/test_project.py` enforces it. Keys the
  renderer writes (`mt_outcome`, `mt_font_size`, `mt_max_font`, `mt_layout`) are listed in the test.
- Region fields the agent sets (`style`, `speaker`, `dismissed`, `link`) live in the page document
  (`mangatl/project/model.py` Region) and in `stages.TEXT_STAGE_KEYS`, so they survive re-preparation.
- Fonts `fonts/CC Wild Words` and `fonts/Anime Ace` are licensed, user-supplied and gitignored; the
  studio's default style sheet uses them. Comic Neue and Bangers are bundled (OFL).
- The studio GPU test turns the studio's lettering defaults off (`typesetting.presets={}`, em dash
  and synthetic styles off) so a page can be compared pixel for pixel with the harness output.
- skia's bundled fontconfig prints "Fontconfig error" lines at start; entry points call
  `mangatl.typeset.fontconfig.start_fontconfig_quietly()` first. Never pipe fd 2 through a Python
  thread (it deadlocks on the GIL).
- Reference outputs: `eval/runs/baseline` (the harness baseline with its LLM cache) and
  `eval/runs/e11def` (a full check run); a new harness (#32) must reproduce them exactly.
- The MCP SDK is v2 (`mcp.server.mcpserver.MCPServer`, not FastMCP); sync tools run in worker threads.

## Layout (the parts that stay)
- `mangatl/studio/`: workspace.py (projects, drafts, renders with outcomes, claims, reviews, glossary,
  script, export), tools.py (MCP tools), web.py + static/ (inspector UI), playbook.md
- `mangatl/pipeline/stages.py` (the stage API), `stored.py` (stages on a stored page)
- `mangatl/engine/`: models, VRAM budget, priority GPU slot, jobs
- `mangatl/project/`: page document, op log with undo, the SQLite project file
- `mangatl/typeset/`: layout, rendering, `style.py` (per-region lettering)
- `mangatl/detect/`, `ocr/`, `clean/`
