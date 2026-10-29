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

## Pipeline

| Stage | Model (local) |
|---|---|
| Bubble detection | YOLOv8 manga109 bubble seg + RT-DETR conjoined-bubble detector + manga109 panel YOLO (reading order) |
| OCR | Speech bubbles: `manga-ocr` (Japanese) / PaddleOCR-VL 1.6 (other languages). Text outside bubbles: Hayai OCR v2.5 Nova, line by line using comic-text-detector lines for multi-line blocks (see below) |
| Cleaning | Flat fill inside bubbles, Koharu SAM-TS-L glyph masks, LaMa-manga inpainting |
| Translation | Gemma 4 26B-A4B QAT (MoE, experts in system RAM) via `llama-server` (llama.cpp, CUDA), thinking off, 3 pages of story context + auto glossary |
| Typesetting | MangaTranslator renderer: per-page font size matching, hyphenation, markdown italics/bold for emphasis |

The engine is [MangaTranslator](https://github.com/meangrinch/MangaTranslator) (`./MangaTranslator`, untouched upstream checkout);
`translate.py` starts/stops llama-server, picks local defaults and forwards everything else.

## Measured (RTX 4070 Ti 12 GB, 10 OpenMantra pages, ~110 bubbles)

| Translator | Peak VRAM (whole pipeline, incl. ~1.4 GB desktop) | Speed | Notes |
|---|---|---|---|
| `--model 26b` (default) | 6.1 GB | ~10 s/page | fewest meaning errors, most natural English |
| `--model 12b` | 11.1 GB | ~6 s/page | ~3-4 % lines with meaning errors (subject mix-ups, idioms) |
| `--model 12b --ocr llm` (vision) | 11.9 GB | ~8 s/page | **not recommended**: Gemma misreads Japanese more than manga-ocr (wrong names, failed crops) |

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

## Common options

```bash
--source Korean --target English     # languages (OCR + reading direction picked automatically)
--glossary vol1_en/glossary.txt      # reuse names/terms from an earlier run (e.g. the previous volume)
--instructions "Keep honorifics"     # extra instructions for the translator
--font "Comic Neue"                  # any font pack folder in ./fonts (drop in e.g. CC Wild Words)
--inpaint flux_klein_4b              # diffusion inpainting for text over artwork (needs more VRAM)
--ocr llm                            # let Gemma read the bubbles itself (vision) — see notes
--osb on|off|auto                    # text outside bubbles (narration/SFX) — see below
--offline                            # hard-forbid network access (all models must be downloaded)
--skip-existing                      # resume: only translate pages without an output yet
--adult                              # 18+ content: translate explicit lines fully (web UI: checkbox)
--keep-other-language                # don't erase other-script blocks (e.g. Chinese next to Japanese)
--no-ocr-correction                  # don't let the LLM repair lines the OCR obviously misread
--llm-url http://host:port/v1 --llm-model-name X   # use another OpenAI-compatible server
--model 12b                          # faster dense translator, fully on GPU
--gguf models/llm/other.gguf [--cpu-moe N]         # use another GGUF (MoE: N layers' experts in RAM)
--vertical-text                      # allow stacking short text vertically (default only for CJK targets)
--upstream-help                      # every MangaTranslator option; unknown flags are passed through
```

## Text outside speech bubbles (OSB)

Narration and chat text printed on the art often comes in large blocks of many vertical lines.
manga-ocr reads a region by squashing it into 224×224 px, which turns such a block into a few pixels
per glyph and makes it invent text. So, like manga-image-translator and BallonsTranslator,
these blocks are read line by line: comic-text-detector (page-level line detector) finds the lines,
Hayai OCR v2.5 Nova (aspect-ratio preserving manga OCR, JA/ZH/KO/EN) reads each one, and they are
joined in reading order. Regions with fewer than 2 detected lines (sound effects, short asides) are
read whole by Hayai, which was trained on onomatopoeia (manga-ocr misread most stylized SFX in tests).
Speech bubbles keep manga-ocr (Hayai and manga-ocr tied on the benchmark bubbles: 1.68% CER). This lives in `osb_ocr.py`, installed into MangaTranslator
at runtime by `mt_upstream.py` (the submodule stays untouched).

Pages that print a Chinese translation next to the Japanese (common in `[中国翻訳]` releases): the
Chinese block is erased and not translated, so the English appears once (`--keep-other-language`
turns this off).

Narration boxes / text drawn on the art need the gated `deepghs/AnimeText_yolo` model.
Accept its terms at <https://huggingface.co/deepghs/AnimeText_yolo> with the account whose token is in
`~/.cache/huggingface/token` (or set `HF_TOKEN`). With `--osb auto` (default) it switches on by itself
once the model is accessible/downloaded.

## Layout

```
translate.py          wrapper / entry point (CLI)
mt_upstream.py        runs MangaTranslator's CLI with the patches below installed
osb_ocr.py            line-by-line OCR for text outside bubbles, full-res crops, other-language erase
text_lines.py         comic-text-detector (ONNX, CPU) line detection
hayai_ocr.py          Hayai OCR v2.5 Nova loader (code vendored in vendor/hayai, audited, pinned)
webui.py, webui/      browser UI + reader (state in webui_data/, default output in translated/)
MangaTranslator/      upstream pipeline (git pull to update, then `uv sync`)
bin/llama.cpp/        prebuilt llama.cpp b11972, CUDA 13.4
models/llm/           translator GGUFs
models/…              detection / OCR / inpainting weights (auto-downloaded)
fonts/                Comic Neue (dialogue), Bangers (OSB/SFX) — both SIL OFL
                      Add your own as fonts/<Name>/ with -Regular/-Bold/-Italic in the file names
                      (e.g. CC Wild Words, Anime Ace; git-ignored, as their licenses forbid redistribution)
                      (model licenses: Hayai Apache-2.0; comic-text-detector weights GPL-3.0)
eval/                 OpenMantra benchmark pages used to check quality
logs/llama-server.log
```
