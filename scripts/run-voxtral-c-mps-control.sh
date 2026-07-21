#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${CUTTLEDOC_VOXTRAL_C_SOURCE_DIR:-/private/tmp/cuttledoc-voxtral-c-134d366}
model_dir=${CUTTLEDOC_VOXTRAL_C_MODEL_DIR:-/tmp/cuttledoc-voxtral-realtime-4b-bf16}
fixture=${CUTTLEDOC_VOXTRAL_PCM_FIXTURE:-/tmp/cuttledoc-audiobook-pilot/de-135_82_000105.f32le}
build_dir=${CUTTLEDOC_VOXTRAL_C_BUILD_DIR:-/private/tmp/cuttledoc-voxtral-c-control-build}
expected_source_revision=134d366c24d20c64b614a3dcc8bda2a6922d077d

if [[ ! -d "$source_dir/.git" ]]; then
  echo "missing voxtral.c checkout: $source_dir" >&2
  exit 1
fi
actual_source_revision=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual_source_revision" != "$expected_source_revision" ]]; then
  echo "expected voxtral.c $expected_source_revision, got $actual_source_revision" >&2
  exit 1
fi
if [[ ! -f "$fixture" ]]; then
  echo "missing mono 16-kHz float32 fixture: $fixture" >&2
  exit 1
fi

bash "$root/scripts/fetch-voxtral-c-mps-model.sh" "$model_dir"
mkdir -p "$build_dir"
make -C "$source_dir" mps

clang \
  -Wall \
  -Wextra \
  -O3 \
  -I "$source_dir" \
  "$root/spikes/voxtral-realtime-c-mps-control/probe.c" \
  "$source_dir/voxtral.mps.o" \
  "$source_dir/voxtral_kernels.mps.o" \
  "$source_dir/voxtral_audio.mps.o" \
  "$source_dir/voxtral_encoder.mps.o" \
  "$source_dir/voxtral_decoder.mps.o" \
  "$source_dir/voxtral_tokenizer.mps.o" \
  "$source_dir/voxtral_safetensors.mps.o" \
  "$source_dir/voxtral_mic_macos.mps.o" \
  "$source_dir/voxtral_metal.o" \
  -framework Accelerate \
  -framework Metal \
  -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph \
  -framework Foundation \
  -framework AudioToolbox \
  -framework CoreFoundation \
  -o "$build_dir/voxtral-c-mps-lifecycle"

ffmpeg \
  -loglevel error \
  -y \
  -f f32le \
  -ar 16000 \
  -ac 1 \
  -i "$fixture" \
  -c:a pcm_s16le \
  "$build_dir/fixture.wav"

echo "LIFECYCLE"
"$build_dir/voxtral-c-mps-lifecycle" "$model_dir"
echo "TRANSCRIPTION"
/usr/bin/time -l "$source_dir/voxtral" \
  -d "$model_dir" \
  -i "$build_dir/fixture.wav"
echo "ARTIFACTS"
echo "source_revision=$actual_source_revision"
stat -f 'voxtral_binary_bytes=%z' "$source_dir/voxtral"
stat -f 'lifecycle_probe_bytes=%z' "$build_dir/voxtral-c-mps-lifecycle"
