#!/usr/bin/env bash
# Doujin CG test set (eval/djn/in, not in git): same flags each run, temperature 0.
# Usage: eval/djn/run.sh <label> [translate.py args...]
set -u
cd "$(dirname "$0")/../.."
label=$1; shift
out=eval/djn/out/$label; mkdir -p "$out"
start=$(date +%s)
MT_LLM_CACHE="${MT_LLM_CACHE:-$PWD/eval/djn/llm_cache}" uv run python translate.py --input eval/djn/in --output "$out" --port 8090 --adult \
  --font "CC Wild Words" --temperature 0 --verbose "$@" > "$out.log" 2>&1
echo "$label rc=$? $(grep -h 'Batch complete' "$out.log") total $(( $(date +%s) - start ))s"
