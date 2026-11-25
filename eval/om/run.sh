#!/usr/bin/env bash
# OpenMantra book (eval/openmantra, CC BY-NC 4.0, not in git) with its professional English.
# Usage: eval/om/run.sh <label> <book> [translate.py args...]   e.g. eval/om/run.sh base tencho_isoro
# --llm-fresh makes temperature-0 translations repeatable.
set -u
cd "$(dirname "$0")/../.."
label=$1; book=$2; shift 2
out=eval/om/out/$label/$book; mkdir -p "$out"
start=$(date +%s)
uv run python translate.py --input eval/openmantra/images/$book/ja --output "$out" --port 8090 --llm-fresh \
  --llm-cache "${LLM_CACHE:-$PWD/$out/llm}" --dump-text "$PWD/$out/text" \
  --font "CC Wild Words" --set translation.temperature=0 --verbose "$@" > "$out.log" 2>&1
echo "$label/$book rc=$? $(grep -h 'Batch complete' "$out.log") total $(( $(date +%s) - start ))s"
