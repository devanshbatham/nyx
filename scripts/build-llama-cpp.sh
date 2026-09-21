#!/usr/bin/env bash
set -euo pipefail

readonly LLAMA_CPP_REVISION=e6cef8152f6e8351a870d8e1a98627139c9c379a
readonly DESTINATION="${1:-/opt/llama.cpp}"
readonly REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

hip_toolchain_available() {
  command -v hipcc >/dev/null 2>&1 || return 1
  command -v hipconfig >/dev/null 2>&1 || return 1

  local root
  local -a roots=()
  [[ -n "${ROCM_PATH:-}" ]] && roots+=("$ROCM_PATH")
  [[ -n "${HIP_PATH:-}" ]] && roots+=("$HIP_PATH")
  roots+=("$(hipconfig --rocmpath 2>/dev/null || true)")
  roots+=("$(hipconfig --path 2>/dev/null || true)")
  roots+=(/opt/rocm)

  for root in "${roots[@]}"; do
    [[ -n "$root" ]] || continue
    if [[ -f "$root/lib/cmake/hip-lang/hip-lang-config.cmake" ||
          -f "$root/lib64/cmake/hip-lang/hip-lang-config.cmake" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -e "$DESTINATION" ]]; then
  echo "Destination already exists: $DESTINATION" >&2
  exit 1
fi

accelerator="${NYX_ACCELERATOR:-auto}"
if [[ "$accelerator" == auto ]]; then
  if [[ "$(uname -s)" == Darwin ]]; then
    accelerator=metal
  elif command -v nvcc >/dev/null 2>&1; then
    accelerator=cuda
  elif hip_toolchain_available; then
    accelerator=hip
  else
    accelerator=cpu
  fi
fi
case "$accelerator" in
  cpu) ;;
  metal) ;;
  cuda) ;;
  hip)
    if ! hip_toolchain_available; then
      echo "NYX_ACCELERATOR=hip requires a complete ROCm installation with hip-lang-config.cmake" >&2
      exit 1
    fi
    ;;
  *) echo "NYX_ACCELERATOR must be auto, cpu, metal, cuda, or hip" >&2; exit 1 ;;
esac

git init "$DESTINATION"
git -C "$DESTINATION" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "$DESTINATION" fetch --depth 1 origin "$LLAMA_CPP_REVISION"
git -C "$DESTINATION" checkout --detach FETCH_HEAD
git -C "$DESTINATION" apply "$REPOSITORY_ROOT/deploy/llama.cpp-grammar-probs.patch"

cmake_args=(-S "$DESTINATION" -B "$DESTINATION/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON)
case "$accelerator" in
  cpu) ;;
  metal) cmake_args+=(-DGGML_METAL=ON) ;;
  cuda) cmake_args+=(-DGGML_CUDA=ON) ;;
  hip) cmake_args+=(-DGGML_HIP=ON) ;;
esac
echo "Building llama.cpp with accelerator: $accelerator"
cmake "${cmake_args[@]}"
cmake --build "$DESTINATION/build" --target llama-server --config Release -j "${NYX_BUILD_JOBS:-4}"
