#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$repo_root/spikes/voxtral-tts-mlx-reference/model-manifest.json"
model_dir=${1:-/tmp/cuttledoc-voxtral-tts-4b-mlx-4bit}
repository=mlx-community/Voxtral-4B-TTS-2603-mlx-4bit
revision=f98fc91b9cb5adc7dab56102c690458276c14c6a

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

actual_snapshot_bytes=$(
  node -e '
    const { statSync } = require("node:fs");
    const { join } = require("node:path");
    const manifest = require(process.argv[1]);
    const modelDir = process.argv[2];
    console.log(
      manifest.artifacts.reduce(
        (bytes, artifact) =>
          bytes + statSync(join(modelDir, artifact.path)).size,
        0,
      ),
    );
  ' "$manifest" "$model_dir"
)
expected_snapshot_bytes=$(
  node -e 'console.log(require(process.argv[1]).conversion.snapshot_bytes)' \
    "$manifest"
)
if [[ "$actual_snapshot_bytes" != "$expected_snapshot_bytes" ]]; then
  echo "Snapshot size mismatch: expected $expected_snapshot_bytes, got $actual_snapshot_bytes" >&2
  exit 1
fi

echo "model_revision=$revision"
echo "model_directory=$model_dir"
echo "snapshot_bytes=$actual_snapshot_bytes"
