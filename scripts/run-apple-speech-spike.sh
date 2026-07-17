#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
source="$root/spikes/apple-speech-shim/Sources/CuttledocSpeechShim.swift"
rust_source="$root/spikes/apple-speech-shim/rust/main.rs"
build_dir=$(mktemp -d /private/tmp/cuttledoc-speech-spike.XXXXXX)
trap 'rm -rf "$build_dir"' EXIT

swiftc \
  -emit-library \
  -emit-module \
  -module-name CuttledocSpeechShim \
  "$source" \
  -o "$build_dir/libcuttledoc_speech_shim.dylib"

rustc \
  "$rust_source" \
  -L "native=$build_dir" \
  -l dylib=cuttledoc_speech_shim \
  -o "$build_dir/cuttledoc-speech-spike"

fixture="$build_dir/fixture.aiff"
say -v Samantha -o "$fixture" "Cuttledoc is testing offline speech transcription on Apple Silicon."

DYLD_LIBRARY_PATH="$build_dir" "$build_dir/cuttledoc-speech-spike" "$fixture"
