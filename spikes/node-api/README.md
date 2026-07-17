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
`--ignore-scripts` path.

This is deliberately one platform artifact. A production release still needs a
package matrix, provenance/signing, Node 22 and 24 CI, background Promise work,
progress/error conversion, and a loader strategy for all supported targets.
