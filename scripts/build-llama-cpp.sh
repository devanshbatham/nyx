#!/usr/bin/env bash
set -euo pipefail

readonly LLAMA_CPP_REVISION=e6cef8152f6e8351a870d8e1a98627139c9c379a
readonly DESTINATION="${1:-/opt/llama.cpp}"
readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -e "$DESTINATION" ]]; then
  echo "Destination already exists: $DESTINATION" >&2
  exit 1
fi

git clone https://github.com/ggml-org/llama.cpp.git "$DESTINATION"
git -C "$DESTINATION" checkout "$LLAMA_CPP_REVISION"
git -C "$DESTINATION" apply "$REPOSITORY_ROOT/deploy/llama.cpp-grammar-probs.patch"

cmake_args=(-S "$DESTINATION" -B "$DESTINATION/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON)
accelerator="${NYX_ACCELERATOR:-auto}"
if [[ "$accelerator" == auto ]]; then
  if [[ "$(uname -s)" == Darwin ]]; then
    accelerator=metal
  elif command -v nvcc >/dev/null 2>&1; then
    accelerator=cuda
  elif command -v hipcc >/dev/null 2>&1 || command -v rocminfo >/dev/null 2>&1; then
    accelerator=hip
  else
    accelerator=cpu
  fi
fi
case "$accelerator" in
  cpu) ;;
  metal) cmake_args+=(-DGGML_METAL=ON) ;;
  cuda) cmake_args+=(-DGGML_CUDA=ON) ;;
  hip) cmake_args+=(-DGGML_HIP=ON) ;;
  *) echo "NYX_ACCELERATOR must be auto, cpu, metal, cuda, or hip" >&2; exit 1 ;;
esac
cmake "${cmake_args[@]}"
cmake --build "$DESTINATION/build" --target llama-server --config Release -j "${NYX_BUILD_JOBS:-4}"
