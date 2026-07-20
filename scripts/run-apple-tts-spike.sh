#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
source="$root/spikes/apple-tts-shim/Sources/CuttledocTtsShim.swift"
rust_source="$root/spikes/apple-tts-shim/rust/main.rs"
text_dir=${CUTTLEDOC_TTS_TEXT_DIR:-}
locale=${CUTTLEDOC_TTS_LOCALE:-de-DE}
passage=${CUTTLEDOC_TTS_PASSAGE:-synthetic-de-origin}

if [[ -z "$text_dir" ]]; then
  echo "CUTTLEDOC_TTS_TEXT_DIR must name a materialized synthetic-roundtrip directory" >&2
  exit 2
fi
text_path="$text_dir/$passage.txt"
if [[ ! -f "$text_path" ]]; then
  echo "missing materialized passage: $text_path" >&2
  exit 2
fi

if [[ -n "${CUTTLEDOC_TTS_BUILD_DIR:-}" ]]; then
  build_dir=$CUTTLEDOC_TTS_BUILD_DIR
  mkdir -p "$build_dir"
else
  build_dir=$(mktemp -d /private/tmp/cuttledoc-tts-spike.XXXXXX)
  trap 'rm -rf "$build_dir"' EXIT
fi
mkdir -p "$build_dir/module-cache"

CLANG_MODULE_CACHE_PATH="$build_dir/module-cache" xcrun swiftc \
  -module-cache-path "$build_dir/module-cache" \
  -emit-library \
  -emit-module \
  -module-name CuttledocTtsShim \
  "$source" \
  -o "$build_dir/libcuttledoc_tts_shim.dylib"

rustc \
  --edition 2021 \
  "$rust_source" \
  -L "native=$build_dir" \
  -l dylib=cuttledoc_tts_shim \
  -o "$build_dir/cuttledoc-apple-tts-spike"

echo "VOICE_INVENTORY"
DYLD_LIBRARY_PATH="$build_dir" \
  "$build_dir/cuttledoc-apple-tts-spike" inventory "$locale"

echo "SYNTHESIS"
echo "locale=$locale"
echo "passage=$passage"
DYLD_LIBRARY_PATH="$build_dir" \
  /usr/bin/time -l \
  "$build_dir/cuttledoc-apple-tts-spike" \
  synthesize \
  "$text_path" \
  "$build_dir/$passage.f32le" \
  --locale "$locale"

echo "CANCELLATION"
DYLD_LIBRARY_PATH="$build_dir" \
  "$build_dir/cuttledoc-apple-tts-spike" \
  cancel \
  "$text_path" \
  --locale "$locale"

echo "ARTIFACTS"
stat -f 'swift_dylib_bytes=%z' "$build_dir/libcuttledoc_tts_shim.dylib"
stat -f 'rust_executable_bytes=%z' "$build_dir/cuttledoc-apple-tts-spike"
stat -f 'audio_bytes=%z' "$build_dir/$passage.f32le"
