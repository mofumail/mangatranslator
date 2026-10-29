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
