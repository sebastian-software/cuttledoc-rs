#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
project="$repo_root/spikes/qwen3-tts-mlx-reference"
manifest="$repo_root/spikes/tts-calibration/qwen3-tts-1.7b-voicedesign-bf16.json"
selection="$repo_root/benchmarks/fixtures/synthetic-roundtrip-selection.json"
runner="$repo_root/spikes/tts-calibration/run_qwen_voicedesign.py"
profile=${CUTTLEDOC_TTS_PROFILE:-qwen-de-clear-documentary}
model_dir=${CUTTLEDOC_QWEN3_TTS_VOICEDESIGN_MODEL_DIR:-/tmp/cuttledoc-qwen3-tts-1.7b-voicedesign-mlx-bf16}
text_dir=${CUTTLEDOC_TTS_TEXT_DIR:-/tmp/cuttledoc-synthetic-roundtrip-passages-4}
output_dir=${CUTTLEDOC_QWEN3_TTS_VOICEDESIGN_OUTPUT_DIR:-/tmp/cuttledoc-qwen3-tts-voicedesign-calibration}
source_revision=$(git -C "$repo_root" rev-parse HEAD)

if [[ ! -d "$model_dir" ]]; then
  echo "Model directory not found: $model_dir" >&2
  echo "Run scripts/fetch-tts-calibration-model.sh first." >&2
  exit 1
fi

passage_id=$(
  node -e '
    const manifest = require(process.argv[1]);
    const profile = manifest.calibration.profiles.find(
      (item) => item.id === process.argv[2],
    );
    if (!profile) process.exit(1);
    console.log(profile.passage_id);
  ' "$manifest" "$profile"
)
text_file="$text_dir/$passage_id.txt"
if [[ ! -f "$text_file" ]]; then
  echo "Materialized passage not found: $text_file" >&2
  echo "Run scripts/materialize-synthetic-roundtrip.mjs first." >&2
  exit 1
fi

uv run \
  --project "$project" \
  --frozen \
  --python /opt/homebrew/bin/python3.12 \
  python "$runner" \
  --manifest "$manifest" \
  --selection "$selection" \
  --profile "$profile" \
  --model-dir "$model_dir" \
  --text-file "$text_file" \
  --output-dir "$output_dir" \
  --source-revision "$source_revision"
