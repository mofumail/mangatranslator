# Evaluation pages

`in/` holds 10 pages from the [OpenMantra dataset](https://github.com/mantra-inc/open-mantra-dataset)
(Hinami et al., "Towards Fully Automated Manga Translation", AAAI 2021), with `annotation.json`
containing the professional English/Chinese reference translations.
OpenMantra is licensed **CC BY-NC 4.0** — non-commercial use only, keep this attribution.

Re-run the benchmark:

```bash
for s in tojime_no_siora balloon_dream tencho_isoro; do
  uv run python translate.py --input eval/in/$s --output eval/out_26b/$s
done
```

## Regression harness

Every refactor is checked with `eval/harness.py`: it runs the eval sets and shows exactly what changed
against a saved baseline.

```bash
uv run python eval/harness.py baseline          # live LLM, ~45 pages; ends with a self-check replay
uv run python eval/harness.py check             # after a change: replay, no LLM server needed
uv run python eval/harness.py check --live      # same, but cache misses are asked to the LLM
uv run python eval/harness.py compare A B       # report on two existing runs
```

- Sets: `om10` (the pages in `in/`, always available), `ch6` and `djn` (local only, skipped when missing).
  Each chapter is its own `translate.py` run with default settings plus `--temperature 0 --format png`.
- A baseline keeps the output pages, a per-page text dump (box, OCR, translation of every region) and every
  LLM answer. `check` replays those answers: changes to detection, OCR, cleaning or typesetting show up as
  image and text differences; a change to what the translator is asked (prompt, OCR text, context) shows up
  as a **cache miss**, and that page's translation fails instead of calling an LLM. The first miss of a
  chapter is the cause; later ones usually follow from it (context and glossary carry it forward).
- The report (`eval/runs/<run>/report.md`) lists only what changed: per page, the changed-pixel share with a
  side-by-side image (baseline | run | changes in red), which regions the changes fall in, and OCR/translation
  differences per region. Exit code 0 means no differences.
- `meta.json` in each run records the git commit, whether the tree was dirty, the exact command per chapter
  and the model files with their sizes.
- Runs live in `eval/runs/` (git-ignored). Take a new baseline whenever a change is intended.
