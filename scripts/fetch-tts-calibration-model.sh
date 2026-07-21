#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest=${1:-}
model_dir=${2:-}

if [[ -z "$manifest" ]] || [[ -z "$model_dir" ]]; then
  echo "usage: bash scripts/fetch-tts-calibration-model.sh MANIFEST OUTPUT_DIR" >&2
  exit 1
fi
if [[ "$manifest" != /* ]]; then
  manifest="$repo_root/$manifest"
fi
if [[ ! -f "$manifest" ]]; then
  echo "Manifest not found: $manifest" >&2
  exit 1
fi

repository=$(node -e 'console.log(require(process.argv[1]).artifact.repository)' "$manifest")
revision=$(node -e 'console.log(require(process.argv[1]).artifact.revision)' "$manifest")
mkdir -p "$model_dir"

file_size() {
  wc -c < "$1" | tr -d ' '
}

file_sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

while IFS=$'\t' read -r path encoded_path expected_bytes expected_sha256; do
  destination="$model_dir/$path"
  partial="$destination.partial"
  mkdir -p "$(dirname "$destination")"

  actual_bytes=0
  actual_sha256=
  if [[ -f "$destination" ]]; then
    actual_bytes=$(file_size "$destination")
    actual_sha256=$(file_sha256 "$destination")
  fi

  if [[ "$actual_bytes" != "$expected_bytes" ]] ||
    [[ "$actual_sha256" != "$expected_sha256" ]]; then
    curl \
      --fail \
      --location \
      --retry 3 \
      --continue-at - \
      --output "$partial" \
      "https://huggingface.co/$repository/resolve/$revision/$encoded_path"

    actual_bytes=$(file_size "$partial")
    actual_sha256=$(file_sha256 "$partial")
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
    for (const artifact of manifest.artifact.artifacts) {
      const encodedPath = artifact.path
        .split("/")
        .map(encodeURIComponent)
        .join("/");
      process.stdout.write(
        `${artifact.path}\t${encodedPath}\t${artifact.bytes}\t${artifact.sha256}\n`,
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
      manifest.artifact.artifacts.reduce(
        (bytes, artifact) =>
          bytes + statSync(join(modelDir, artifact.path)).size,
        0,
      ),
    );
  ' "$manifest" "$model_dir"
)
expected_snapshot_bytes=$(
  node -e 'console.log(require(process.argv[1]).artifact.snapshot_bytes)' \
    "$manifest"
)
if [[ "$actual_snapshot_bytes" != "$expected_snapshot_bytes" ]]; then
  echo "Snapshot size mismatch: expected $expected_snapshot_bytes, got $actual_snapshot_bytes" >&2
  exit 1
fi

echo "model_revision=$revision"
echo "model_directory=$model_dir"
echo "snapshot_bytes=$actual_snapshot_bytes"
