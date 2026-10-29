#!/usr/bin/env bash
# Translation model bake-off on the OpenMantra pages: same OCR/prompt/flags, temperature 0.
# Usage: eval/bake/run.sh <label> [translate.py model args...]
set -u
cd "$(dirname "$0")/../.."
label=$1; shift
out=eval/bake/out/$label; mkdir -p "$out"
start=$(date +%s)
for s in tojime_no_siora balloon_dream tencho_isoro; do
  uv run python translate.py --input eval/in/$s --output "$out/$s" --port 8090 \
    --temperature 0 --verbose "$@" > "$out/$s.log" 2>&1
  echo "$label $s rc=$? $(grep -h 'Batch complete' "$out/$s.log")"
done
echo "$label total $(( $(date +%s) - start ))s"
