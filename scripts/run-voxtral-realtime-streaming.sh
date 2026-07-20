#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
project="$repo_root/spikes/voxtral-realtime-mlx-reference"
model_dir=${CUTTLEDOC_VOXTRAL_REALTIME_MODEL_DIR:-/tmp/cuttledoc-voxtral-realtime-4b-mlx-4bit}
manifest=${CUTTLEDOC_VOXTRAL_STREAMING_MANIFEST:-$repo_root/benchmarks/fixtures/audiobook-pilot.json}
fixture_dir=${CUTTLEDOC_VOXTRAL_STREAMING_FIXTURE_DIR:-/tmp/cuttledoc-audiobook-pilot}
fixture_id=${CUTTLEDOC_VOXTRAL_STREAMING_FIXTURE_ID:-audiobook-de-135_82_000105}
output=${CUTTLEDOC_VOXTRAL_STREAMING_OUTPUT:-/tmp/cuttledoc-voxtral-realtime-streaming.json}
repetitions=${CUTTLEDOC_VOXTRAL_STREAMING_REPETITIONS:-2}

if [[ ! -d "$model_dir" ]]; then
  echo "Model directory not found: $model_dir" >&2
  echo "Run scripts/fetch-voxtral-realtime-mlx-model.sh first." >&2
  exit 1
fi

"$project/.venv/bin/python" \
  "$project/stream_probe.py" \
  "$model_dir" \
  "$manifest" \
  "$fixture_dir" \
  "$fixture_id" \
  --delay-ms 480 \
  --delay-ms 2400 \
  --chunk-ms 80 \
  --max-decode-tokens 16 \
  --max-tokens 4096 \
  --repetitions "$repetitions" \
  > "$output"

echo "Wrote $output"
