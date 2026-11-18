#!/usr/bin/env bash
# OpenMantra book (eval/openmantra, CC BY-NC 4.0, not in git) with its professional English.
# Usage: eval/om/run.sh <label> <book> [translate.py args...]   e.g. eval/om/run.sh base tencho_isoro
# MT_LLM_FRESH=1 (default) makes temperature-0 translations repeatable.
set -u
cd "$(dirname "$0")/../.."
label=$1; book=$2; shift 2
out=eval/om/out/$label/$book; mkdir -p "$out"
start=$(date +%s)
MT_DUMP_TEXT="$PWD/$out/text" MT_LLM_FRESH="${MT_LLM_FRESH:-1}" MT_LLM_CACHE="${MT_LLM_CACHE:-$PWD/$out/llm}" \
  uv run python translate.py --input eval/openmantra/images/$book/ja --output "$out" --port 8090 \
  --font "CC Wild Words" --temperature 0 --verbose "$@" > "$out.log" 2>&1
echo "$label/$book rc=$? $(grep -h 'Batch complete' "$out.log") total $(( $(date +%s) - start ))s"
