# Node-API packaging spike

This spike exercises issue #9 as an installable native package rather than as a
source-level binding demo. The native Rust class reads the shared stream
contract fixture at compile time; JavaScript adds only `AsyncIterator` lifecycle
semantics.

Run on Apple Silicon:

```sh
scripts/run-node-api-spike.sh
```

The script builds the `cdylib`, renames it to the platform-specific `.node`
artifact, runs `npm pack`, installs the resulting tarball into clean temporary
ESM and CommonJS consumers, verifies early cancellation, and rejects leaked
Rust sources or Cargo files in the npm package. The ESM install keeps lifecycle
scripts enabled while compiler guards make any Cargo, CMake, C/C++, or
`node-gyp` fallback fail; the CommonJS install also proves the
`--ignore-scripts` path and the explicit missing-artifact diagnostic.

The packed consumers also submit copied PCM bytes to real Rust work on the
libuv worker pool. The test keeps the JavaScript event loop live, verifies four
native progress callbacks, preserves the pre-mutation checksum, converts a
Rust failure into a rejected Promise, and cooperatively cancels work through an
`AbortSignal`.

This is deliberately one platform artifact. A production release still needs a
complete package matrix, provenance/signing, and a loader strategy for all
supported targets. CI runs this final-tarball test on Node 22 and 24 using a
GitHub-hosted macOS arm64 runner.
