#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
project="$repo_root/spikes/text-generation-mlx-reference"
manifest=${CUTTLEDOC_TEXT_GENERATION_MANIFEST:-$project/model-manifest.json}

IFS=$'\t' read -r manifest_id fixture_relative prompt_relative < <(
  node -e '
    const manifest = require(process.argv[1]);
    process.stdout.write([
      manifest.id,
      manifest.result_contract.fixture_path,
      manifest.generation_contract.prompt_path,
    ].join("\t") + "\n");
  ' "$manifest"
)
if [[ -z "$manifest_id" ]] || [[ -z "$fixture_relative" ]] || [[ -z "$prompt_relative" ]]; then
  echo "Manifest did not provide model id, fixture, and prompt paths: $manifest" >&2
  exit 1
fi
fixture=${CUTTLEDOC_TEXT_GENERATION_FIXTURE:-$repo_root/$fixture_relative}
prompt=${CUTTLEDOC_TEXT_GENERATION_PROMPT:-$repo_root/$prompt_relative}
model_dir=${CUTTLEDOC_TEXT_GENERATION_MODEL_DIR:-/tmp/cuttledoc-$manifest_id}

output=${CUTTLEDOC_TEXT_GENERATION_OUTPUT:-/tmp/cuttledoc-text-generation-mlx-reference.json}
source_revision=$(git -C "$repo_root" rev-parse HEAD)

if [[ ! -d "$model_dir" ]]; then
  echo "Model directory not found: $model_dir" >&2
  echo "Run scripts/fetch-text-generation-mlx-model.sh first with the same manifest." >&2
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
