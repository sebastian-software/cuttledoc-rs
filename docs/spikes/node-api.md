# Node-API packaging spike (#9)

Status: implementation complete; locally verified on Node 24 / macOS arm64;
Node 22 and 24 packed-artifact CI added.

## What was exercised

- a `napi-rs` `cdylib` with N-API 8 as the minimum ABI,
- a Rust-owned stream handle compiled from the same contract vectors as the
  Rust reducer,
- an `AsyncIterableIterator` wrapper with deterministic `return()` cleanup,
- one tarball exposing the same API through ESM and CommonJS,
- clean-install tests with lifecycle scripts enabled and disabled, including
  compiler guards for the enabled path, and
- an artifact-content check that excludes Cargo metadata, native build files,
  and Rust sources;
- real Rust work on the libuv worker pool with a copied PCM input, progress,
  Promise rejection, and cooperative `AbortSignal` cancellation; and
- deliberate diagnostics for unsupported platforms and missing native
  artifacts.

The package intentionally contains only the macOS arm64 `.node` binary, two
loaders, declarations, and package metadata. Product logic remains on the Rust
side of the boundary.

## Local result

The packed artifact installed and executed successfully in isolated ESM and
CommonJS consumers on Node 24.18.0. The enabled-lifecycle install had Cargo,
CMake, C/C++, and `node-gyp` guards on `PATH` and invoked none of them. Both
consumers received sequences 1, 2, and 3 from `volatile_to_final`; both also
closed an iterator early through `return()`. Replace/revoke operations and
their affected ranges survive the packed boundary. The background-work check
reports progress at 25/50/75/100%, retains the checksum of PCM mutated by the
caller after submission, leaves the JavaScript event loop responsive, maps a
Rust failure to a rejected Promise, and rejects cancelled work.

Node 22 is part of the intended support floor but was not installed on this
machine. The repository workflow runs the exact same build-pack-install-test
script on Node 22 and 24; its first green run is required before closing #9.

## Decision

The thin Node-API shape is viable. Continue with per-target artifact generation
and keep JS limited to loading, type declarations, and idiomatic iterator
lifecycle. Do not duplicate reducer or engine logic in TypeScript.
