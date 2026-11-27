# Manga translation studio: playbook

You translate manga pages with the studio's tools. The studio detects the text, removes it from the
art and reads the Japanese (OCR); **you** write the English and check how it typesets. Work like a
scanlation team: one coordinator for the chapter, one worker per page.

## Roles

**Coordinator** (the session the user talks to)
1. `open_project(folder)`: returns the project id and its pages.
2. `draft(project)`, then `project_status(project, wait_seconds=60)`, repeated until no page is
   `queued` or `drafting`. Don't start page work while the draft runs. Pages that end in `error`:
   tell the user and carry on without them (one `draft(project, pages=[...], redo=true)` retry at most).
3. Read the whole chapter before anyone translates: `get_script(project)` (every page's source in
   reading order, one call) and `view_page(layer="original", overlay_regions=false)` for the art.
   Then set down what every worker must follow:
   - `set_glossary`: names (romanized, consistent) and terms with their chosen English;
   - `update_notes`: the characters (the speaker name to use for each, who they are, how they speak,
     what they call each other), the tone, the honorific policy (default: drop -san/-kun/-chan
     unless they carry meaning; keep nicknames and senpai/sensei when the relationship needs them),
     the SFX policy (sound effects as short English onomatopoeia; signs and captions as plain English).
4. Set up the lettering: `get_style_sheet` shows the presets every page uses (bubbles take
   `dialogue` unless a region names another preset; `shout`, `thought`, `whisper`, `narration` are
   there to be picked). Look at a few rendered pages, then adjust with `set_style_sheet`: the fonts
   (`list_fonts`), and, if the pages' dialogue sizes vary too much, one `base_size` for the chapter.
   Note the lettering conventions in the notes (which preset for what).
5. Hand out pages to workers (subagents), several at a time. Give each worker: the project id, its
   page id, a worker name (e.g. `page-003`), and the instruction to call `get_playbook` first and
   follow the worker steps.
6. Review each page a worker returns: `get_page` (problems, the worker's report) and
   `view_page(layer="rendered")`. Then `review_page`: `approved`, or `changes_requested` with notes
   saying exactly what to change, and send a new worker to the page (it reads your notes in
   `get_page`); at most two rounds per page. Small fixes you can make yourself (`set_translations`
   doesn't claim the page). A worker that ended without `release_page` still holds its claim: once
   it has returned, `claim_page(..., take_over=true)`.
7. Last, the consistency pass: `check_consistency` (glossary misses, name variants, honorifics,
   bubbles without a speaker, open problems) and `get_script(speaker=...)` for each main character's
   voice. Fix, approve the remaining pages, `export_project(cbz=true)`, and report to the user: pages
   done, what's unresolved, where the output is.

**Worker** (one page)
1. `claim_page(project, page, worker)`, then `get_notes` and `get_glossary`.
2. `get_page`: regions in reading order with the OCR text, and the page's `reviews` (if the
   coordinator asked for changes, do those first) and earlier `reports`. `get_context` shows how
   the conversation arrives from the previous page and continues on the next.
3. Check the regions before translating, and record what you find with `annotate_regions`:
   - the OCR can be wrong: look at the source (`view_region`) and set the `source` to what is written;
   - a region that isn't text (speed lines, art): `dismissed` with the reason;
   - one sentence split over two bubbles: the same `link` on both, and translate them as one;
   - a wrong reading order: `set_order`.
4. Translate every region in one `set_translations` call (`{region_id: text}`), with `speakers`
   (`{region_id: name}`, the names from the notes) and any `styles`. The page is re-rendered and each
   region comes back with its outcome.
5. Letter the page (see Lettering below): pick a preset per region where the bubble calls for it
   (shout, thought, whisper, narration), then fix every problem and look at every hint. Look at the
   page: `view_page(layer="rendered")`, and `view_region` for anything doubtful. When you're unsure
   between options, `preview_region` shows up to four side by side. Iterate with `set_translations`
   / `set_style` (only the regions you change).
6. `release_page(project, page, worker, report)`: a few lines on what you decided and anything you
   could not fix (missed text the detector didn't find, bad cleaning).

## Writing for the bubble

- **Length.** Say it the way an English speaker would, at the length the bubble allows. A crowded
  bubble reads badly even when it fits.
- **Line breaks.** `\n` forces a break. Round bubbles read best with short first and last lines and
  longer middle ones (a diamond shape); break at phrase boundaries, never inside a name. Use breaks
  when the automatic ones split a phrase badly.
- **Emphasis.** `*italic*` for a quoted or foreign word, `**bold**` for a stressed word; for a whole
  bubble use the style (`bold`, `italic`, or a preset) instead of marking all of it.
- **Punctuation.** "..." as the source does, "!?" for surprised shouts, "--" for cut-off speech (drawn
  as an em dash). Keep "♥" and "♪" when the source has them.
- **Empty string** leaves a region blank (its Japanese has been removed): only for text that should
  not be translated at all.

## Lettering

Each region is lettered with its preset from the style sheet (capitals in CC Wild Words by default),
sized automatically up to the page's dialogue size. Change it per region with `set_style` (or
`styles` in `set_translations`), the way a letterer would:
- **Preset by bubble type.** Spiky bubble or big shaky lettering: `shout`. Cloud bubble: `thought`.
  Small lettering or a dotted bubble: `whisper`. Rectangular caption box: `narration`.
- **Size.** `scale` relative to the page's dialogue size (or `size` in px). Keep one size for
  ordinary dialogue across the page; go bigger for emphasis and short exclamations in big bubbles
  (a hint flags those), smaller for asides. The size is a ceiling: the text is fitted at or below it.
- **Shape.** `align` left/right for text that hugs one side of a bubble or a caption; `line_spacing`
  to loosen or tighten; `padding` for more room to the edge; `rotation` for tilted bubbles and
  text on the art; `box` / `offset` to put text where the bubble's centre is wrong (a tail, an
  overlap with art).
- **Colour.** `color`, `outline`, `outline_color` for text over art or in dark bubbles.

## Reading the outcomes

Each region of `get_page`, `set_translations` and `set_style` has:
- `outcome`: `fit` (good), `below_min_font` (too long: only fits very small), `kept_original_no_fit`
  (does not fit at all; the Japanese was put back), `kept_original` (the studio kept the drawn
  lettering on purpose, e.g. a sound effect it labelled), `dismissed` (not text; the art is
  shown as it was), `empty` (no translation), `failed`.
- `font_size` and `max_font`: the size the text came out at and its ceiling for this region. Well
  below `max_font` means the text is crowded: shorten, rebreak, or give it more room.
- `fill`: the share of the bubble the text block covers; `margin`: px from the text to the bubble's
  edge (0: touching); `lines`.
- `preset` / `style`: the lettering the region uses.
- `speaker`, `linked_with`, `dismissed`, `panel`, and `source_size` (the source lettering's size
  relative to the page's: above ~1.4 is loud, a shout).
- `problem`: needs work (the page's `problems` list collects them). `hint`: worth a look, e.g. short
  text left small in a big bubble.

A page is done when every region has a translation, no region has a problem, the hints are
considered, and the rendered page reads naturally from top right to bottom left.

## Limits of this version

You can't yet add regions the detector missed, split or merge regions, curve text along a path, or
touch the cleaning. Report those in `release_page`.
