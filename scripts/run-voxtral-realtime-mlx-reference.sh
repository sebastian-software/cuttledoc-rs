#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
project="$repo_root/spikes/voxtral-realtime-mlx-reference"
model_dir=${CUTTLEDOC_VOXTRAL_REALTIME_MODEL_DIR:-/tmp/cuttledoc-voxtral-realtime-4b-mlx-4bit}
fixture_dir=${CUTTLEDOC_VOXTRAL_REALTIME_FIXTURE_DIR:-/tmp/cuttledoc-audiobook-pilot}
manifest=${CUTTLEDOC_VOXTRAL_REALTIME_MANIFEST:-$repo_root/benchmarks/fixtures/audiobook-pilot.json}
delay_ms=${CUTTLEDOC_VOXTRAL_REALTIME_DELAY_MS:-480}
output=${CUTTLEDOC_VOXTRAL_REALTIME_OUTPUT:-/tmp/cuttledoc-voxtral-realtime-${delay_ms}ms.json}

if [[ ! -d "$model_dir" ]]; then
  echo "Model directory not found: $model_dir" >&2
  echo "Run scripts/fetch-voxtral-realtime-mlx-model.sh first." >&2
  exit 1
fi

CUTTLEDOC_VOXTRAL_REALTIME_PYTHON="$project/.venv/bin/python" \
CUTTLEDOC_VOXTRAL_REALTIME_MODEL_DIR="$model_dir" \
  node "$repo_root/scripts/run-phase0-asr-matrix.mjs" \
    --candidate voxtral-realtime-mlx-reference \
    --manifest "$manifest" \
    --fixture-dir "$fixture_dir" \
    --transcription-delay-ms "$delay_ms" \
    --repetitions 2 \
    --output "$output"
