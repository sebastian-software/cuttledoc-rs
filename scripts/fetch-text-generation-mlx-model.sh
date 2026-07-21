#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$repo_root/spikes/text-generation-mlx-reference/model-manifest.json"
model_dir=${1:-/tmp/cuttledoc-qwen3-0.6b-4bit}
repository=mlx-community/Qwen3-0.6B-4bit
revision=73e3e38d981303bc594367cd910ea6eb48349da8

mkdir -p "$model_dir"

while IFS=$'\t' read -r path expected_bytes expected_sha256; do
  destination="$model_dir/$path"
  partial="$destination.partial"
  mkdir -p "$(dirname "$destination")"

  actual_bytes=0
  actual_sha256=
  if [[ -f "$destination" ]]; then
    actual_bytes=$(stat -f '%z' "$destination")
    actual_sha256=$(shasum -a 256 "$destination" | awk '{print $1}')
  fi

  if [[ "$actual_bytes" != "$expected_bytes" ]] ||
    [[ "$actual_sha256" != "$expected_sha256" ]]; then
    curl \
      --fail \
      --location \
      --retry 3 \
      --continue-at - \
      --output "$partial" \
      "https://huggingface.co/$repository/resolve/$revision/$path"

    actual_bytes=$(stat -f '%z' "$partial")
    actual_sha256=$(shasum -a 256 "$partial" | awk '{print $1}')
    if [[ "$actual_bytes" != "$expected_bytes" ]] ||
      [[ "$actual_sha256" != "$expected_sha256" ]]; then
      echo "Artifact verification failed: $path" >&2
      echo "Expected: $expected_bytes bytes, SHA-256 $expected_sha256" >&2
      echo "Actual:   $actual_bytes bytes, SHA-256 $actual_sha256" >&2
      exit 1
    fi
    mv "$partial" "$destination"
  fi

  echo "verified $path ($expected_bytes bytes)"
done < <(
  node -e '
    const manifest = require(process.argv[1]);
    for (const artifact of manifest.artifacts) {
      process.stdout.write(
        `${artifact.path}\t${artifact.bytes}\t${artifact.sha256}\n`,
      );
    }
  ' "$manifest"
)

echo "model_revision=$revision"
echo "model_directory=$model_dir"
