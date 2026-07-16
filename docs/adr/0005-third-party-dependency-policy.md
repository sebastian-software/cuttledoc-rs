# ADR-0005: Strict third-party dependency acceptance policy

## Status

Accepted

## Context

Apple-local inference often depends on young wrappers and model ports. A project can be technically impressive or recently updated without having sustainable maintainership, stable releases, responsive security handling, sufficient adoption, or current platform CI.

Cuttledoc is especially exposed to native runtime, build, and packaging dependencies. An abandoned wrapper on the critical path would force an unplanned fork or block Apple OS, Rust, Node, and upstream model updates.

## Decision

Keep the production dependency surface deliberately small. Every material runtime, binding, native build component, code generator, and security-sensitive or transitive dependency receives one explicit disposition:

1. **Accepted production dependency** — actively maintained, sufficiently established for its role, bounded, and realistically replaceable.
2. **Repository-owned boundary** — a deliberately small adapter or FFI layer over an established platform or upstream library that this repository agrees to maintain.
3. **Reference only** — useful as ideas, tests, compatibility information, or prior art, but absent from production manifests, build scripts, generated release artifacts, and stable APIs.
4. **Rejected** — unsuitable even as a source because of licensing, provenance, security, or correctness concerns.

Assessment considers meaningful commits and releases, maintainer responsiveness, bus factor and governance, downstream production use, current platform CI, compatibility discipline, security response, license/provenance, transitive and binary cost, and replaceability. No single metric is sufficient. In particular, a recent commit or successful technical spike does not qualify a dependency by itself.

A fork is not a loophole. Forking or vendoring requires a separate accepted ADR naming the maintenance owner, adopted code and license, upstream tracking and security process, supported CI, expected divergence, and exit plan.

## Consequences

### Positive

- Reduces supply-chain and abandonment risk.
- Prevents experiments from silently becoming permanent architecture.
- Encourages small replaceable interop boundaries.
- Makes the true maintenance cost of a fork visible before adoption.

### Negative

- Some convenient Rust wrappers will remain reference-only.
- Cuttledoc may need to own a small FFI layer or defer a runtime entirely.
- Evaluations require maintenance evidence in addition to benchmarks.

## Initial application

- `mlx-rs`, OminiX-MLX, and similar young Rust MLX projects are reference-only until the Phase 0 audit demonstrates that they pass this policy.
- `mlx-node` is prior art only and cannot become a dependency because it retains Node/C++ ownership and install-time native build concerns.
- `objc2-core-ml`, `napi-rs`, whisper.cpp integration options, FFmpeg/audio choices, `mistral.rs`, and Candle must each receive an explicit disposition before production adoption.

## Validation

Phase 0 produces `docs/dependency-policy.md` and a dependency disposition table for every recommended runtime path. CI later enforces accepted licenses, advisories, and the reviewed dependency inventory.
