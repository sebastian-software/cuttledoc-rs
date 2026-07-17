#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
native_dir="$repo_root/spikes/node-api/native"
package_source="$repo_root/spikes/node-api/package"
tests_source="$repo_root/spikes/node-api/tests"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "node-api spike requires darwin-arm64" >&2
  exit 1
fi

cargo build --manifest-path "$native_dir/Cargo.toml" --release --locked

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/cuttledoc-node-api.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
export npm_config_cache="$work_dir/npm-cache"

package_dir="$work_dir/package"
mkdir -p "$package_dir"
cp "$package_source/package.json" "$package_dir/package.json"
cp "$package_source/index.cjs" "$package_dir/index.cjs"
cp "$package_source/index.js" "$package_dir/index.js"
cp "$package_source/index.d.ts" "$package_dir/index.d.ts"
cp "$native_dir/target/release/libcuttledoc_node_spike.dylib" \
  "$package_dir/cuttledoc-node-spike.darwin-arm64.node"

tarball_name="$(cd "$package_dir" && npm pack --silent --pack-destination "$work_dir")"
tarball_path="$work_dir/$tarball_name"

archive_entries="$(tar -tzf "$tarball_path")"
grep -q 'package/cuttledoc-node-spike.darwin-arm64.node' <<<"$archive_entries"
grep -q 'package/index.cjs' <<<"$archive_entries"
grep -q 'package/index.js' <<<"$archive_entries"
if grep -Eq 'Cargo|src/' <<<"$archive_entries"; then
  echo "npm artifact unexpectedly contains Rust build inputs" >&2
  exit 1
fi

compiler_guard_dir="$work_dir/compiler-guard"
mkdir -p "$compiler_guard_dir"
for tool in cargo cc c++ clang cmake node-gyp; do
  ln -s "$tests_source/compiler-guard.sh" "$compiler_guard_dir/$tool"
done

for mode in esm commonjs; do
  app_dir="$work_dir/$mode-app"
  mkdir -p "$app_dir"
  if [[ "$mode" == "esm" ]]; then
    PATH="$compiler_guard_dir:$PATH" \
      npm_config_node_gyp="$compiler_guard_dir/node-gyp" \
      npm install --prefix "$app_dir" --no-audit --no-fund "$tarball_path"
    cp "$tests_source/esm.mjs" "$app_dir/contract-test.mjs"
    node "$app_dir/contract-test.mjs"
  else
    npm install --prefix "$app_dir" --ignore-scripts --no-audit --no-fund "$tarball_path"
    cp "$tests_source/commonjs.cjs" "$app_dir/contract-test.cjs"
    node "$app_dir/contract-test.cjs"
  fi
done

artifact_size="$(stat -f '%z' "$tarball_path")"
echo "npm artifact: $tarball_name ($artifact_size bytes)"
