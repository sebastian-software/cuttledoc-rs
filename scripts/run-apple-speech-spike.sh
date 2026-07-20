#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
source="$root/spikes/apple-speech-shim/Sources/CuttledocSpeechShim.swift"
rust_source="$root/spikes/apple-speech-shim/rust/main.rs"
locale=${CUTTLEDOC_SPEECH_LOCALE:-en-US}
if [[ -n "${CUTTLEDOC_SPEECH_BUILD_DIR:-}" ]]; then
  build_dir=$CUTTLEDOC_SPEECH_BUILD_DIR
  mkdir -p "$build_dir"
else
  build_dir=$(mktemp -d /private/tmp/cuttledoc-speech-spike.XXXXXX)
  trap 'rm -rf "$build_dir"' EXIT
fi

swiftc \
  -emit-library \
  -emit-module \
  -module-name CuttledocSpeechShim \
  "$source" \
  -o "$build_dir/libcuttledoc_speech_shim.dylib"

rustc \
  --edition 2021 \
  "$rust_source" \
  -L "native=$build_dir" \
  -l dylib=cuttledoc_speech_shim \
  -o "$build_dir/cuttledoc-speech-spike"

source_fixture="${CUTTLEDOC_SPEECH_FIXTURE:-$root/../cuttledoc/packages/cuttledoc/fixtures/fleurs-en-000.ogg}"
if [[ ! -f "$source_fixture" ]]; then
  echo "missing real fixture: $source_fixture" >&2
  exit 2
fi
fixture="$build_dir/fleurs-en-000.f32le"
ffmpeg \
  -y \
  -v error \
  -i "$source_fixture" \
  -ar 16000 \
  -ac 1 \
  -f f32le \
  -acodec pcm_f32le \
  "$fixture"

echo "IDENTITY"
codesign -d --verbose=4 "$build_dir/cuttledoc-speech-spike" 2>&1
if otool -l "$build_dir/cuttledoc-speech-spike" | grep -q __info_plist; then
  echo "embedded_info_plist=true"
else
  echo "embedded_info_plist=false"
fi

echo "STREAM"
echo "locale=$locale"
DYLD_LIBRARY_PATH="$build_dir" \
  /usr/bin/time -l \
  "$build_dir/cuttledoc-speech-spike" \
  "$fixture" \
  --locale "$locale"

if [[ "${CUTTLEDOC_SPEECH_SKIP_CANCEL:-0}" != "1" ]]; then
  echo "CANCEL"
  DYLD_LIBRARY_PATH="$build_dir" \
    "$build_dir/cuttledoc-speech-spike" \
    "$fixture" \
    --locale es-ES \
    --cancel
fi

echo "ARTIFACTS"
stat -f 'swift_dylib_bytes=%z' "$build_dir/libcuttledoc_speech_shim.dylib"
stat -f 'rust_executable_bytes=%z' "$build_dir/cuttledoc-speech-spike"
