#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${CUTTLEDOC_MLX_SOURCE_DIR:-/private/tmp/cuttledoc-mlx-v0.32.0}
model_dir=${CUTTLEDOC_VOXTRAL_MODEL_DIR:-/tmp/cuttledoc-voxtral-realtime-4b-mlx-4bit}
fixture=${CUTTLEDOC_VOXTRAL_PCM_FIXTURE:-/tmp/cuttledoc-audiobook-pilot/de-135_82_000105.f32le}
build_dir=${CUTTLEDOC_VOXTRAL_BUILD_DIR:-/private/tmp/cuttledoc-voxtral-mlx-direct-build}
build_home=${CUTTLEDOC_VOXTRAL_BUILD_HOME:-/private/tmp/cuttledoc-mlx-build-home}
expected_mlx_commit=7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247
frontend_oracle="$root/benchmarks/oracles/voxtral-realtime.audiobook-de-135_82_000105.frontend-480ms.json"

if [[ ! -d "$source_dir/.git" ]]; then
  echo "missing official MLX checkout: $source_dir" >&2
  exit 1
fi
actual_mlx_commit=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual_mlx_commit" != "$expected_mlx_commit" ]]; then
  echo "expected official MLX v0.32.0 at $expected_mlx_commit, got $actual_mlx_commit" >&2
  exit 1
fi
if ! xcrun --find metal >/dev/null; then
  echo "Xcode Metal Toolchain is required" >&2
  exit 1
fi
if [[ ! -f "$fixture" ]]; then
  echo "missing mono 16-kHz float32 fixture: $fixture" >&2
  exit 1
fi

bash "$root/scripts/fetch-voxtral-realtime-mlx-model.sh" "$model_dir"
mkdir -p "$build_dir" "$build_home" "$build_home/clang-module-cache"

env \
  HOME="$build_home" \
  CLANG_MODULE_CACHE_PATH="$build_home/clang-module-cache" \
  cmake \
  -S "$root/spikes/mlx-direct" \
  -B "$build_dir" \
  -DMLX_SOURCE_DIR="$source_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
env \
  HOME="$build_home" \
  CLANG_MODULE_CACHE_PATH="$build_home/clang-module-cache" \
  cmake --build "$build_dir" --target cuttledoc_voxtral_mlx_shim --parallel 2

rustc \
  --edition 2024 \
  -C opt-level=3 \
  -C strip=symbols \
  -C 'link-arg=-Wl,-rpath,@executable_path' \
  "$root/spikes/voxtral-realtime-mlx-direct/rust/main.rs" \
  -L "native=$build_dir" \
  -l dylib=cuttledoc_voxtral_mlx_shim \
  -o "$build_dir/cuttledoc-voxtral-mlx"

echo "MODEL"
"$build_dir/cuttledoc-voxtral-mlx" inspect "$model_dir"
echo "FRONTEND"
frontend_result="$build_dir/voxtral-frontend-480ms.json"
"$build_dir/cuttledoc-voxtral-mlx" \
  frontend "$model_dir" "$fixture" 480 gpu >"$frontend_result"
cat "$frontend_result"
node "$root/scripts/validate-voxtral-mlx-frontend.mjs" \
  --oracle "$frontend_oracle" \
  --actual "$frontend_result"
echo "CONTRACT"
"$build_dir/cuttledoc-voxtral-mlx" contract "$model_dir" "$fixture" gpu
echo "ARTIFACTS"
echo "mlx_commit=$actual_mlx_commit"
stat -f 'shim_dylib_bytes=%z' "$build_dir/libcuttledoc_voxtral_mlx_shim.dylib"
stat -f 'rust_executable_bytes=%z' "$build_dir/cuttledoc-voxtral-mlx"
stat -f 'metallib_bytes=%z' "$build_dir/mlx.metallib"
stat -f 'model_safetensors_bytes=%z' "$model_dir/model.safetensors"
