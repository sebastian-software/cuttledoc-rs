#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
source_dir=${CUTTLEDOC_MLX_SOURCE_DIR:?set CUTTLEDOC_MLX_SOURCE_DIR to MLX v0.32.0}
expected_commit=7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247
actual_commit=$(git -C "$source_dir" rev-parse HEAD)

if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "expected MLX v0.32.0 commit $expected_commit, got $actual_commit" >&2
  exit 1
fi

if ! xcrun --find metal >/dev/null; then
  echo "Xcode Metal Toolchain is required; install it with xcodebuild -downloadComponent MetalToolchain" >&2
  exit 1
fi

build_dir=$(mktemp -d /private/tmp/cuttledoc-mlx-direct.XXXXXX)
trap 'rm -rf "$build_dir"' EXIT

cmake \
  -S "$root/spikes/mlx-direct" \
  -B "$build_dir" \
  -DMLX_SOURCE_DIR="$source_dir"
cmake --build "$build_dir" --target cuttledoc_mlx_shim --parallel 2

rustc \
  --edition 2024 \
  "$root/spikes/mlx-direct/rust/main.rs" \
  -L "native=$build_dir" \
  -l dylib=cuttledoc_mlx_shim \
  -o "$build_dir/cuttledoc-mlx-direct-spike"

DYLD_LIBRARY_PATH="$build_dir" "$build_dir/cuttledoc-mlx-direct-spike"
