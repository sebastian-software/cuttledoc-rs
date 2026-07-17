# Node-API packaging spike (#9)

Status: implemented; locally verified on Node 24 / macOS arm64.

## What was exercised

- a `napi-rs` `cdylib` with N-API 8 as the minimum ABI,
- a Rust-owned stream handle compiled from the same contract vectors as the
  Rust reducer,
- an `AsyncIterableIterator` wrapper with deterministic `return()` cleanup,
- one tarball exposing the same API through ESM and CommonJS,
- clean-install tests with lifecycle scripts enabled and disabled, including
  compiler guards for the enabled path, and
- an artifact-content check that excludes Cargo metadata and Rust sources.

The package intentionally contains only the macOS arm64 `.node` binary, two
loaders, declarations, and package metadata. Product logic remains on the Rust
side of the boundary.

## Local result

The packed artifact installed and executed successfully in isolated ESM and
CommonJS consumers on Node 24.18.0. The enabled-lifecycle install had Cargo,
CMake, C/C++, and `node-gyp` guards on `PATH` and invoked none of them. Both
consumers received sequences 1, 2, and 3 from `volatile_to_final`; both also
closed an iterator early through `return()`.

Node 22 is part of the intended support floor but was not installed on this
machine. Its result therefore remains a CI-matrix task rather than a claimed
local pass.

## Decision

The thin Node-API shape is viable. Continue with per-target artifact generation
and keep JS limited to loading, type declarations, and idiomatic iterator
lifecycle. Do not duplicate reducer or engine logic in TypeScript.
