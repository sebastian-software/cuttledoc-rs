# ADR-0007: Apple Silicon and macOS 26 baseline for v3

## Status

Accepted

## Context

Cuttledoc 2 nominally supports Intel macOS, but the core value proposition — local Apple-optimized ASR — has always been Apple Silicon only. Keeping an `x86_64-apple-darwin` cloud-only target in v3 would cost a permanent build target, CI lane, and release-test surface for a discontinued platform.

Apple's SpeechAnalyzer/SpeechTranscriber APIs require macOS 26 (Tahoe) and have no Objective-C or C interface: they are Swift-only. Supporting older macOS versions would force a permanent weak-linking dual path in which the system Speech backend is an optional extra rather than a first-class bakeoff candidate. Modern CoreML capabilities relevant to this product — the thread-safe async `prediction(from:)` API, stateful models (`MLState`), and current compute-unit options — are also unconditionally available once the deployment target is high enough.

Cuttledoc 2 remains a released, working product for users on older systems.

## Decision

- Cuttledoc v3 supports macOS only on Apple Silicon (`aarch64-apple-darwin`) with a macOS 26 deployment target.
- macOS Intel is not supported by v3 at all, including cloud-only transcription. Cuttledoc 2 remains the product for Intel Macs and for macOS releases before 26.
- The Phase 7 migration plan must define an explicit Cuttledoc 2 support window (at minimum critical fixes) before v3 replaces it.
- Apple SpeechAnalyzer/SpeechTranscriber is promoted from "system baseline measurement" to a full bakeoff candidate, reached through a repository-owned Swift shim (ADR-0005 decision class 2). The shim must handle the AssetInventory bundle-identifier requirement for CLI and library consumers.
- Linux and Windows cloud-only support is unchanged.

## Consequences

### Positive

- One macOS build target and one deployment baseline; no weak linking or `#available` dual paths.
- SpeechAnalyzer competes on equal footing in the Phase 0 bakeoff, including its streaming-first result model.
- Async CoreML prediction, stateful models, and current Metal features are usable unconditionally.
- Smaller compatibility, CI, and release matrices.

### Negative

- Users on macOS 14/15 (Sonoma/Sequoia) cannot use v3 until they upgrade; they stay on Cuttledoc 2.
- The release build for the Speech shim requires a Swift toolchain in addition to Rust.
- Self-hosted Apple Silicon runners must run macOS 26 to exercise the Speech backend.

## Alternatives considered

### macOS 14/15 baseline with weak-linked Speech

Keeps a broader install base but makes the system Speech backend a permanently optional capability, doubles the local test matrix, and blocks stateful CoreML models on the oldest supported OS.

### Keep Intel macOS as cloud-only

Pays a permanent target, CI, and support tax for a shrinking platform on which the differentiating local features can never work.

## Validation

The Phase 0 SpeechAnalyzer spike must prove the Swift shim from a Rust-owned lifecycle, including asset installation, timestamps, confidence, and volatile/final result handling. The compatibility matrix and PLAN.md reflect this baseline.
