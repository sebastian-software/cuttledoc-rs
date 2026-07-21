#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
project="$repo_root/spikes/text-generation-mlx-reference"
manifest="$project/model-manifest.json"
fixture="$repo_root/benchmarks/postprocessing/fixtures/issue7-de-audiobook-whisper.json"
prompt="$repo_root/benchmarks/postprocessing/prompts/surface-only-v1.txt"
model_dir=${CUTTLEDOC_TEXT_GENERATION_MODEL_DIR:-/tmp/cuttledoc-qwen3-0.6b-4bit}
output=${CUTTLEDOC_TEXT_GENERATION_OUTPUT:-/tmp/cuttledoc-text-generation-mlx-reference.json}
source_revision=$(git -C "$repo_root" rev-parse HEAD)

if [[ ! -d "$model_dir" ]]; then
  echo "Model directory not found: $model_dir" >&2
  echo "Run scripts/fetch-text-generation-mlx-model.sh first." >&2
  exit 1
fi

uv run \
  --project "$project" \
  --frozen \
  --python /opt/homebrew/bin/python3.12 \
  python "$project/run_reference.py" \
  --manifest "$manifest" \
  --model-dir "$model_dir" \
  --fixture "$fixture" \
  --prompt "$prompt" \
  --output "$output" \
  --source-revision "$source_revision"
