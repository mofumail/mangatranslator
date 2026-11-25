#!/usr/bin/env bash
# Reference chapter (eval/ch6/jp, pro scanlation in eval/ch6/en; not in git).
# Usage: eval/ch6/run.sh <label> [translate.py args...]
# --llm-fresh makes temperature-0 translations repeatable; --llm-cache <dir> stores and replays
# responses (point LLM_CACHE at another run's llm/ for typesetting-only A/B runs).
set -u
cd "$(dirname "$0")/../.."
label=$1; shift
out=eval/ch6/out/$label; mkdir -p "$out"
start=$(date +%s)
uv run python translate.py --input eval/ch6/jp --output "$out" --port 8090 --llm-fresh \
  --llm-cache "${LLM_CACHE:-$PWD/$out/llm}" --dump-text "$PWD/$out/text" \
  --font "CC Wild Words" --set translation.temperature=0 --verbose "$@" > "$out.log" 2>&1
echo "$label rc=$? $(grep -h 'Batch complete' "$out.log") total $(( $(date +%s) - start ))s"
