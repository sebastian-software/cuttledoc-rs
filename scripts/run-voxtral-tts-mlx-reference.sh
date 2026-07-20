#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
project="$repo_root/spikes/voxtral-tts-mlx-reference"
manifest="$project/model-manifest.json"
model_dir=${CUTTLEDOC_VOXTRAL_TTS_MODEL_DIR:-/tmp/cuttledoc-voxtral-tts-4b-mlx-4bit}
text_dir=${CUTTLEDOC_TTS_TEXT_DIR:-/tmp/cuttledoc-synthetic-roundtrip-pilot-1-verified}
output_dir=${CUTTLEDOC_VOXTRAL_TTS_OUTPUT_DIR:-/tmp/cuttledoc-voxtral-tts-reference-run}
source_revision=$(git -C "$repo_root" rev-parse HEAD)

if [[ ! -d "$model_dir" ]]; then
  echo "Model directory not found: $model_dir" >&2
  echo "Run scripts/fetch-voxtral-tts-mlx-model.sh first." >&2
  exit 1
fi
if [[ ! -f "$text_dir/synthetic-de-origin.txt" ]]; then
  echo "Materialized passage not found: $text_dir/synthetic-de-origin.txt" >&2
  echo "Run scripts/materialize-synthetic-roundtrip.mjs first." >&2
  exit 1
fi

uv run \
  --project "$project" \
  --frozen \
  --python /opt/homebrew/bin/python3.12 \
  python "$project/run_reference.py" \
  --manifest "$manifest" \
  --model-dir "$model_dir" \
  --text-file "$text_dir/synthetic-de-origin.txt" \
  --output-dir "$output_dir" \
  --source-revision "$source_revision"
