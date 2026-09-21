#!/usr/bin/env bash
set -euo pipefail

: "${NYX_GGUF_PATH:?Set NYX_GGUF_PATH to nyx.gguf}"

exec "${LLAMA_SERVER_BIN:-llama-server}" \
  --model "$NYX_GGUF_PATH" \
  --host "${NYX_LLAMACPP_HOST:-127.0.0.1}" \
  --port "${NYX_LLAMACPP_PORT:-30000}" \
  --threads "${NYX_CPU_THREADS:-8}" \
  --threads-batch "${NYX_CPU_THREADS:-8}" \
  --ctx-size 4096 \
  --parallel 1 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --cache-ram 0 \
  --n-gpu-layers all \
  --metrics \
  --no-webui
