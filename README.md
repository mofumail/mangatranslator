# Local manga translator

Fully local, batch Japanese → English (or any pair) manga translation.
Nothing leaves the machine at inference time; models are downloaded once on first use.

```bash
uv run python translate.py --input chapter01/ --output chapter01_en/
```

## Web UI

```bash
uv run python webui.py        # open http://127.0.0.1:7860
```

- Drop a folder or `.zip`/`.cbz` on the page (uploaded as a copy), or pick one **on this PC** (used in place).
- Choose the output folder (default `./translated/<name>`), languages and translator, then Translate.
- The built-in reader opens right away and fills in pages as they finish (originals shown until then).
  Page mode (right-to-left by default, click the sides or use arrow keys), scroll mode (`M`),
  hold-to-compare original (`O`), page grid (`G`), chapter picker, fullscreen (`F`), `?` for all keys.
- The library keeps every run: cancel, resume (only missing pages are redone), download as zip,
  or "Open folder in reader" for any folder of images (e.g. CLI output).
- One job runs at a time (single GPU); others queue. Bound to 127.0.0.1 only; nothing loads from the internet.

## Command line

Input can be a folder of images, a volume folder with one subfolder per chapter (translated in natural
order, output mirrors the folder structure), a single image, or a `.zip`/`.cbz`.
Output pages are written as `<name>_translated.png`, plus a `glossary.txt` of names/terms
(e.g. `相川仁 -> Jin Aikawa`) that the translator keeps reusing so names stay consistent.

Every run also writes `project.mtl` to the output folder: one file (SQLite) holding each page's
source, regions (with stable ids), OCR reads, translations, cleaning and typesetting inputs, and
the history of every change, which the editor will open. Copy it to share or resume a project;
`--set output.project=null` turns it off. `python eval/rebuild_check.py <output>` renders every
page from the project alone and compares it with the PNG written (`--export DIR`: each page as JSON).

## Pipeline

| Stage | Model (local) |
|---|---|
| Bubble detection | YOLOv8 manga109 bubble seg + RT-DETR conjoined-bubble detector + manga109 panel YOLO (reading order) |
| OCR | Japanese: Hayai OCR v2.5 Nova everywhere. Speech bubbles: Hayai, cross-checked with `manga-ocr`. Text outside bubbles: line by line using comic-text-detector lines for multi-line blocks (see below). Other languages: PaddleOCR-VL 1.6 |
| Cleaning | Flat fill inside bubbles, Koharu SAM-TS-L glyph masks, LaMa-manga inpainting |
| Translation | Gemma 4 26B-A4B QAT (MoE, experts in system RAM) via `llama-server` (llama.cpp, CUDA), thinking off, 3 pages of story context + auto glossary |
| Typesetting | Skia/HarfBuzz renderer: per-page font size matching, hyphenation, markdown italics/bold for emphasis; ♥ ♪ ★ drawn from a bundled symbol font when the lettering font lacks them; sound effects keep the artist's lettering with a small English label beside it, sized to the effect (`--sfx redraw` to replace them) |

The pipeline lives in `./mangatl`, derived from [MangaTranslator](https://github.com/meangrinch/MangaTranslator)
(Apache-2.0, see `mangatl/NOTICE`); `translate.py` starts/stops llama-server, picks local defaults and forwards everything else.

## Measured (RTX 4070 Ti 12 GB, 10 OpenMantra pages, ~110 bubbles)

| Translator | Peak VRAM (whole pipeline, incl. ~1.4 GB desktop) | Speed | Notes |
|---|---|---|---|
| `--model 26b` (default) | 8.8 GB (6.1 with `--vision off`) | ~12 s/page | fewest meaning errors, most natural English |
| `--model 12b` | 11.1 GB | ~6 s/page | ~3-4 % lines with meaning errors (subject mix-ups, idioms) |

Cleaning (no leftover Japanese in bubbles) and typesetting (fits bubbles, real italics/bold for emphasis) were
clean on every page checked, including colour pages. First page of a run is slower (model warm-up), and
the llama-server start adds ~5-10 s per run.

## Translation quality notes

- Model bake-off on the 10 OpenMantra pages (temperature 0, blind scoring of serious meaning errors):
  Gemma 4 26B-A4B **0**, Qwen3.6-35B-A3B 4, Ministral-3-14B 18 (output shifted into wrong bubbles).
  Re-run with `eval/bake/run.sh <label> <translate.py args>` and `eval/bake/sheet.py`.
- A localization style note is always sent (natural English, contractions, past-tense narration).
  Per-series choices go in `--instructions` / the web UI field, e.g. `Translate お姉ちゃん as "Sis"`.
- If a long source line comes back implausibly short (the model sometimes drops the rest of a
  block), the page is translated once more and the fuller answer kept.
- A same-model "polish" pass was tested and changed little; not enabled.
- Page image for the translator (`--vision`, local OCR kept, ~2 GB VRAM). Showing it for every line
  (`--vision full`) was measured on three sets with repeatable runs: on a shounen chapter checked against
  its English scanlation it fixed OCR misreads, speaker mix-ups and sound effects; on an OpenMantra book it
  added meaning errors and turned a boyish narrator into "she"; on doujin CG pages it dropped the " / "
  line separators of text blocks and rewrote credits. So the default is `--vision short`: translate from
  text, then redo only short items (outside-bubble text up to 10 characters, bubbles up to 5) with the page
  in view. That kept the gains (sound effects, the speaker of a short reply, the chapter number on a title,
  names read from furigana: 和音 Kanon → Kazune, 宏也 Futoshi → Hiroya on a held-out OpenMantra book)
  without touching long lines; ~2-3 s/page more. Restyle-only answers (*Huff* → **HUFF**) are ignored.
  Off by default for `--model 12b` (out of VRAM on 12 GB), `--gguf` and `--llm-url`.
- Tried and removed: running character notes sent with every page (built from text alone, they attributed
  lines to the wrong speakers and changed as much for the worse as for the better), and letting Gemma read
  the bubbles itself (more misreads than local OCR: wrong names, failed crops, "corrected" Japanese such as
  キャンセル -> マンセル).
- Prompt rules on top of the base prompt: honorifics dropped like a professional localization
  (`--honorifics keep` to keep them), Western spelling for katakana Western names (プリシラ → Priscilla),
  a vowel-only bubble that continues the previous word is that word drawn out (死ねェ / エ / エエエ!! →
  Die! / Die... / DIEEEE!!), and sound effects never enter the name glossary.
- Doujin CG test set (17 pages from 8 works, `eval/djn/run.sh <label>`, pages not in git): no pages with
  empty bubbles or dropped hearts; ~17 s/page.

## Common options

```bash
--source Korean --target English     # languages (OCR + reading direction picked automatically)
--glossary vol1_en/glossary.txt      # reuse names/terms from an earlier run (e.g. the previous volume)
--instructions "Keep honorifics"     # extra instructions for the translator
--font "Comic Neue"                  # any font pack folder in ./fonts (drop in e.g. CC Wild Words)
--inpaint opencv|none                # text over artwork: simple fill or leave it (default: LaMa)
--osb on|off|auto                    # text outside bubbles (narration/SFX) — see below
--offline                            # hard-forbid network access (all models must be downloaded)
--skip-existing                      # resume: only translate pages without an output yet
--adult                              # 18+ content: translate explicit lines fully (web UI: checkbox)
--keep-other-language                # don't erase other-script blocks (e.g. Chinese next to Japanese)
--vision short|full|off              # page image for the translator (default short; see notes)
--sfx label|redraw                   # sound effects: keep the art + small label (default), or redraw
--honorifics keep                    # keep -san/-sama/... (default: drop them, as pro localizations do)
--no-ocr-correction                  # don't let the LLM repair lines the OCR obviously misread
--llm-url http://host:port/v1 --llm-model-name X   # use another OpenAI-compatible server
--model 12b                          # faster dense translator, fully on GPU
--gguf models/llm/other.gguf [--cpu-moe N]         # use another GGUF (MoE: N layers' experts in RAM)
--vertical-text                      # allow stacking short text vertically (default only for CJK targets)
--set typesetting.bubbles.max_font_size=28   # any setting (repeatable); see --show-settings
--show-settings                      # print the run's settings as JSON
--verbose                            # detailed pipeline log
```

## Text outside speech bubbles (OSB)

Narration and chat text printed on the art often comes in large blocks of many vertical lines.
manga-ocr reads a region by squashing it into 224×224 px, which turns such a block into a few pixels
per glyph and makes it invent text. So, like manga-image-translator and BallonsTranslator,
these blocks are read line by line: comic-text-detector (page-level line detector) finds the lines,
Hayai OCR v2.5 Nova (aspect-ratio preserving manga OCR, JA/ZH/KO/EN) reads each one, and they are
joined in reading order. Regions with fewer than 2 detected lines (sound effects, short asides) are
read whole by Hayai, which was trained on onomatopoeia (manga-ocr misread most stylized SFX in tests).
Sound effects are read twice (tight crop and with a margin, which fixes most misreads of hand-drawn
kana) and the fuller read kept. Speech bubbles are read by Hayai too: on the OpenMantra bubbles it and
manga-ocr are about equal (0.9% / 0.7% character errors), but manga-ocr turns hand-drawn moans in doujin
bubbles into unrelated words (ぅ -> そして、). When one read contains the other the longer is kept.
This lives in `mangatl/ocr/line_ocr.py` (with `mangatl/detect/text_lines.py` and `mangatl/ocr/hayai.py`).

Pages that print a Chinese translation next to the Japanese (common in `[中国翻訳]` releases): the
Chinese block is erased and not translated, so the English appears once (`--keep-other-language`
turns this off). A kana-free block only counts as Chinese when it uses simplified-only characters, so
all-kanji Japanese labels (status screens like 状態：乳房開発完了) are translated.

Typesetting outside bubbles: long text (narration, dialogue printed on the art) uses the dialogue font,
upright, one quoted line or (thought) per line; lists (stream comments, credits, `Name: text` lines) keep
one entry per line. Short text stays sound-effect style (Bangers, caps, follows the tilt). A bubble whose
translation can't fit even at a smaller size gets its original lettering back instead of staying blank.

Narration boxes / text drawn on the art need the gated `deepghs/AnimeText_yolo` model.
Accept its terms at <https://huggingface.co/deepghs/AnimeText_yolo> with the account whose token is in
`~/.cache/huggingface/token` (or set `HF_TOKEN`). With `--osb auto` (default) it switches on by itself
once the model is accessible/downloaded.

## Development

```bash
uv run --group dev pytest                 # unit + static tests; with models and a harness run of
                                          # om10, also the stage API on a stored page (GPU)
uv run --group dev ruff check . && uv run --group dev ruff format --check .
uv run --group dev pyright                # type check (our own modules; see [tool.pyright])
uv run python eval/harness.py check       # regression check against the saved baseline (GPU)
uv run python eval/rebuild_check.py eval/runs/<run>   # pages rebuilt from project.mtl == PNGs
```

CI (`.github/workflows/ci.yml`) runs ruff, pyright and the tests on CPU with a light set of
packages; tests only import modules whose import chain stays free of torch / cv2. Every change to
the pipeline is checked with the harness (see `eval/README.md`): zero differences unless intended.

## Layout

```
translate.py          wrapper / entry point (CLI)
webui.py, webui/      browser UI + reader (state in webui_data/, default output in translated/)
mangatl/              the pipeline (derived from MangaTranslator at e347a09; see mangatl/NOTICE)
  settings.py         Settings: one typed object (pydantic) describing a run, by stage; written to
                      <output>/settings.json with every run (secrets excluded)
  presets.py          user-facing options (translate.py, web UI) -> Settings
  cli.py              runs one input with a settings file (translate.py runs `python -m mangatl.cli`)
  project/            the page document (model.py), operation log with undo / conflicts / staleness
                      (ops.py), the project file (store.py, SQLite), images as patches and 1-bit
                      masks (blobs.py), pipeline state in and out (capture.py), rebuild.py,
                      region matching for re-detection (match.py)
  pipeline/           stages.py: the stage API (detect, redetect, clean, prepare, ocr, translate,
                      typeset, render; run() composes them), page.py (load a page, run, save),
                      stored.py (stages on a stored page: load, save as operations, story context),
                      long strips, batches
  detect/             bubbles, panels / reading order, outside-text detector, text lines, tilt
  ocr/                Hayai OCR and line-by-line reading (line_ocr.py)
  clean/              bubble cleaning, glyph masks, LaMa inpainting
  translate/          prompts and calls (service.py), vision_pass.py, glossary, LLM client
  typeset/            layout, fonts, renderer, symbols, narration, sound-effect labels
  ml/                 model loading (model_manager.py), RT-DETR adapter, SAM-TS (hi_sam)
  common/             shared helpers: image utils, caching, logging, eval_hooks (harness)
  inputs/             archives and path lists
bin/llama.cpp/        prebuilt llama.cpp b11972, CUDA 13.4
models/llm/           translator GGUFs
models/…              detection / OCR / inpainting weights (auto-downloaded)
assets/symbols/       DejaVu Sans Bold for ♥ ♪ ★ (free license, see LICENSE-DejaVu.txt)
fonts/                Comic Neue (dialogue), Bangers (OSB/SFX) — both SIL OFL
                      Add your own as fonts/<Name>/ with -Regular/-Bold/-Italic in the file names
                      (e.g. CC Wild Words, Anime Ace; git-ignored, as their licenses forbid redistribution)
tests/                unit tests and static checks (imports resolve, Settings reads exist);
                      test_stages_gpu.py: region-scoped stages on a stored page
eval/                 harness.py: regression check of every change against a saved baseline (replays
                      the LLM answers, diffs pages and text; see eval/README.md);
                      OpenMantra benchmark pages used to check quality; eval/djn/ doujin test runner;
                      eval/ch6/ a chapter vs its English scanlation (refs.py reads the pro lettering,
                      compare.py / checklist.py side by side); eval/om/ OpenMantra books vs their
                      pro English. Runs pass --llm-fresh, which makes temperature-0 output repeatable
                      (model licenses: Hayai Apache-2.0; comic-text-detector weights GPL-3.0)
logs/llama-server.log
```
