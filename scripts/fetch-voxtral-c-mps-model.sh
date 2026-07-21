#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$root/spikes/voxtral-realtime-c-mps-control/control-manifest.json"
model_dir=${1:-/tmp/cuttledoc-voxtral-realtime-4b-bf16}
repository=mistralai/Voxtral-Mini-4B-Realtime-2602
revision=2769294da9567371363522aac9bbcfdd19447add

mkdir -p "$model_dir"

while IFS=$'\t' read -r path expected_bytes expected_sha256; do
  destination="$model_dir/$path"
  partial="$destination.partial"
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
    for (const artifact of manifest.model.artifacts) {
      process.stdout.write(
        `${artifact.path}\t${artifact.bytes}\t${artifact.sha256}\n`,
      );
    }
  ' "$manifest"
)

echo "model_revision=$revision"
echo "model_directory=$model_dir"
