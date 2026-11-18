#!/usr/bin/env bash
# Reference chapter (eval/ch6/jp, pro scanlation in eval/ch6/en; not in git).
# Usage: eval/ch6/run.sh <label> [translate.py args...]
# MT_LLM_FRESH=1 (default) makes temperature-0 translations repeatable (see osb_ocr.install_llm_cache);
# MT_LLM_CACHE=<dir> replays recorded responses (for typesetting-only A/B runs).
set -u
cd "$(dirname "$0")/../.."
label=$1; shift
out=eval/ch6/out/$label; mkdir -p "$out"
start=$(date +%s)
MT_DUMP_TEXT="$PWD/$out/text" MT_LLM_FRESH="${MT_LLM_FRESH:-1}" MT_LLM_CACHE="${MT_LLM_CACHE:-$PWD/$out/llm}" uv run python translate.py --input eval/ch6/jp --output "$out" --port 8090 \
  --font "CC Wild Words" --temperature 0 --verbose "$@" > "$out.log" 2>&1
echo "$label rc=$? $(grep -h 'Batch complete' "$out.log") total $(( $(date +%s) - start ))s"
