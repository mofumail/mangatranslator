#!/usr/bin/env bash
# Vision LLM for refs.py: Gemma 4 12B fully on GPU, port ${1:-8091}. Stop with: kill <pid it prints>
cd "$(dirname "$0")/../.."
bin/llama.cpp/llama-server -m models/llm/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  --mmproj models/llm/mmproj-F16.gguf -b 2048 -ub 2048 --host 127.0.0.1 --port "${1:-8091}" \
  -c 8192 -np 1 -ngl 99 -fa on --reasoning off --no-webui > eval/ch6/out/vision_server.log 2>&1 &
echo $!
