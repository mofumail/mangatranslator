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
3. Read the whole chapter before anyone translates: `get_page` for every page (source text in reading
   order) and `view_page(layer="original", overlay=false)` for the art. Then write the chapter notes
   with `update_notes`:
   - characters: name (romanized, consistent), who they are, how they speak (polite, rough, childish,
     archaic), what they call each other;
   - terms and names of places, items, techniques, with the chosen English;
   - tone of the work; honorific policy (default: drop -san/-kun/-chan unless they carry meaning;
     keep nicknames and senpai/sensei when the relationship needs them);
   - SFX policy: outside-bubble text (kind `outside`) is drawn in capitals; translate sound effects
     as short English onomatopoeia, signs and captions as plain English.
4. Set up the lettering: `get_style_sheet` shows the presets every page uses (bubbles take
   `dialogue` unless a region names another preset; `shout`, `thought`, `whisper`, `narration` are
   there to be picked). Look at a few rendered pages, then adjust with `set_style_sheet`: the fonts
   (`list_fonts`), and, if the pages' dialogue sizes vary too much, one `base_size` for the chapter.
   Note the lettering conventions in the chapter notes (which preset for what).
5. Hand out pages to workers (subagents), several at a time. Give each worker: the project id, its
   page id, a worker name (e.g. `page-003`), and the instruction to call `get_playbook` and
   `get_notes` first and follow the worker steps.
6. Review each page a worker returns: `get_page` (problems list) and `view_page(layer="rendered")`.
   Send it back with concrete notes (a new worker, same page) when something reads wrong or looks
   wrong; at most two rounds per page. A worker that ended without `release_page` still holds its
   claim: once it has returned, take the page over with `claim_page(..., take_over=true)` (yourself
   or for the next worker).
7. Last: a consistency pass over all pages (names, terms, each character's voice), fixing with
   `set_translations` yourself, then report to the user: pages done, open problems, the output folder.

**Worker** (one page)
1. `claim_page(project, page, worker)`, then `get_notes(project)`.
2. `get_page`: regions in reading order, with the OCR text. The OCR can be wrong: when a read looks
   off, look at the source with `view_region` or `view_page(layer="original")` and translate what is
   actually written.
3. `get_page` on the previous page too, for the flow of the conversation.
4. Translate every region and send them in one `set_translations` call (`{region_id: text}`). The
   page is re-rendered and each region comes back with its outcome.
5. Letter the page (see Lettering below): pick a preset per region where the bubble calls for it
   (shout, thought, whisper, narration), then fix every problem and look at every hint. Look at the
   page: `view_page(layer="rendered")`, and `view_region` for anything doubtful. When you're unsure
   between options, `preview_region` shows up to four side by side. Iterate with `set_translations`
   / `set_style` (only the regions you change; both take styles).
6. `release_page(project, page, worker, report)`: a few lines on what you decided (names, tone) and
   anything you could not fix (missed text the detector didn't find, bad cleaning, a wrong reading
   order, OCR you couldn't read).

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
  lettering on purpose, e.g. a sound effect it labelled), `empty` (no translation), `failed`.
- `font_size` and `max_font`: the size the text came out at and its ceiling for this region. Well
  below `max_font` means the text is crowded: shorten, rebreak, or give it more room.
- `fill`: the share of the bubble the text block covers; `margin`: px from the text to the bubble's
  edge (0: touching); `lines`.
- `preset` / `style`: the lettering the region uses.
- `problem`: needs work (the page's `problems` list collects them). `hint`: worth a look, e.g. short
  text left small in a big bubble.

A page is done when every region has a translation, no region has a problem, the hints are
considered, and the rendered page reads naturally from top right to bottom left.

## Limits of this version

You can't yet fix the OCR text itself, add or remove regions, change the reading order, curve
text along a path, or touch the cleaning. Report those in `release_page`.
