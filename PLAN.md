# Cuttledoc Rust migration plan

## Objective

Deliver Cuttledoc v3 as a Rust library with a native CLI and thin Node.js bindings while preserving the useful behavior of Cuttledoc 2, `parakeet-coreml`, and `whisper-coreml`.

This is a staged migration, not a line-by-line rewrite. Each phase must produce evidence that lowers a specific risk.

## Definition of done

The incubator is ready to become Cuttledoc v3 when all of the following are true:

- The Rust API and native CLI transcribe supported audio/video inputs through Parakeet, Whisper, and OpenAI.
- The Node API covers the stable Cuttledoc 2 use cases without owning product logic.
- Parakeet retains VAD-based segmentation, timestamps, and the 15-second internal model constraint without exposing that constraint to users.
- Whisper retains `large-v3-turbo`, language detection, timestamps, the CoreML encoder, and the whisper.cpp decoder.
- Model downloads are resumable or atomic, validated, observable, configurable, and compatible with existing caches or an explicit migration command.
- Apple Silicon accuracy and performance stay within the agreed tolerances in `docs/compatibility-matrix.md`.
- The npm tarball installs on a clean supported machine without compiling native code.
- ESM and CommonJS can load the same prebuilt Node-API artifact.
- Rust CLI releases and npm releases are reproducible and smoke-tested from their final artifacts.
- The existing production repository has a documented upgrade, rollback, and deprecation path.

## Non-goals for the first release

- Reimplementing CoreML, whisper.cpp, or FFmpeg in pure Rust.
- Supporting Intel macOS for local CoreML inference.
- Adding new ASR models before parity is established.
- Supporting every possible audio codec without an FFmpeg fallback.
- Porting embedded GGUF post-processing before its product value and distribution cost are revalidated.
- Publishing every workspace crate independently.
- Maintaining the old low-level `parakeet-coreml` and `whisper-coreml` Node APIs indefinitely.

## Phase 0 — Architecture and feasibility

**Purpose:** retire the highest-risk assumptions before scaffolding a large workspace.

Deliverables:

- Accept or revise the initial ADRs.
- Build a minimal CoreML interop spike from Rust on `darwin-arm64`.
- Decide the first bridge for the existing Objective-C++ Parakeet code.
- Decide whether Whisper uses a maintained Rust wrapper or a local `whisper.cpp` sys crate.
- Prove a `napi-rs` addon can load from both ESM and CommonJS.
- Pack and install a minimal npm tarball in an empty directory.
- Record binary size, startup time, model load time, and cleanup behavior.

Exit criteria:

- A real CoreML model is loaded and invoked from a Rust-owned lifecycle.
- The result crosses Rust → Node without a second orchestration layer.
- The packed addon loads on Node 22 and Node 24 without `node-gyp`.
- Interop ownership and thread-affinity rules are documented.

## Phase 1 — Workspace foundation

**Purpose:** create only the structure justified by the spike.

Deliverables:

- Cargo workspace and pinned stable Rust toolchain.
- `cuttledoc`, `cuttledoc-coreml`, `cuttledoc-models`, `cuttledoc-cli`, and `cuttledoc-node` starting crates.
- Formatting, Clippy, unit tests, dependency auditing, license checks, and macOS CI.
- Shared error model with stable error codes.
- Shared progress-event and cancellation types.
- Initial fixture import with provenance metadata.
- Artifact-oriented smoke-test harness.

Exit criteria:

- The empty vertical architecture builds locally and in CI.
- Platform-specific code does not prevent non-macOS crates from compiling.
- Rust and Node APIs are generated or checked from one set of domain types.

## Phase 2 — Parakeet vertical slice

**Purpose:** validate the complete product path with the faster and structurally richer local backend.

Scope:

- Parakeet TDT 0.6B v3 model lifecycle.
- Preprocessor, encoder, decoder, joint decision, and vocabulary handling.
- Silero VAD v6 lifecycle.
- 36 ms VAD frames at 16 kHz.
- Configurable threshold, minimum silence, and minimum speech duration.
- Automatic handling of the fixed 15-second ASR input.
- Structured segments and combined text.
- Engine reuse and deterministic cleanup.
- Existing 25-language capability metadata.
- Model download, validation, force refresh, and progress.
- Rust, CLI, and Node surfaces.

Exit criteria:

- Existing Parakeet fixtures pass agreed text and timestamp tolerances.
- Long audio is segmented without exposing the model's chunk limit.
- A packed npm artifact performs a real transcription on Apple Silicon.
- Repeated engine creation and disposal does not leak materially.

## Phase 3 — Whisper vertical slice

**Purpose:** bring the quality/coverage backend into the same lifecycle.

Scope:

- whisper.cpp pinned initially to the current `v1.8.2` baseline, then deliberately updated.
- `large-v3-turbo` only for initial parity.
- GGML model plus CoreML encoder completeness checks.
- Encoder on CoreML/ANE and decoder through whisper.cpp Metal/CPU behavior.
- Automatic language detection and explicit language selection.
- Segment timestamps and confidence.
- Thread and GPU configuration where it remains meaningful.
- Existing 99-language capability metadata.
- Model downloads, validation, progress, and atomic installation.

Exit criteria:

- Existing Whisper fixtures pass agreed text, language, and timestamp tolerances.
- Model updates are pinned and reproducible rather than implicitly “latest.”
- Node and CLI artifacts load without bundler-specific native-addon workarounds.

## Phase 4 — Audio, OpenAI, and orchestration parity

**Purpose:** make the Rust core a usable replacement for Cuttledoc 2.

Scope:

- Audio/video probing and decode to normalized 16 kHz mono `f32` samples.
- Initial compatibility path through a managed FFmpeg binary or configured system binary.
- Backend selection: explicit backend and `auto` behavior.
- OpenAI transcription models and API-key handling.
- Backend capability reporting.
- Engine caching and cleanup.
- CLI transcription, output paths, plain text, Markdown, and model commands.
- Cancellation and progress propagation across downloads, decode, load, and inference.

Exit criteria:

- Current CLI happy paths are covered by artifact tests.
- Linux and Windows can build the non-CoreML product and use OpenAI.
- Credentials never appear in command lines, logs, or persisted configuration.

## Phase 5 — Transcript enhancement decision

**Purpose:** prevent the existing `@cuttledoc/llm` implementation from silently defining the Rust architecture.

Evaluate independently:

- OpenAI enhancement through the Rust core.
- Ollama enhancement through the Rust core.
- Embedded GGUF inference and its binary/model distribution cost.
- Chunking, correction statistics, Markdown formatting, and prompt compatibility.
- Whether enhancement belongs in the initial v3 or a later minor release.

Gate:

- Keep enhancement only where measured transcript quality, install size, and operational complexity justify it.
- Preserve raw transcription as a complete product regardless of this decision.

## Phase 6 — Distribution and compatibility

**Purpose:** prove that the final delivery mechanisms work, not just source checkouts.

Deliverables:

- Rust crate publication plan.
- Native CLI archives, checksums, signatures, and Homebrew path.
- One user-facing `cuttledoc` npm package.
- Platform-specific npm binary packages only if required as hidden distribution details.
- Node 22/24 test matrix for ESM and CommonJS.
- macOS deployment-target and Apple Silicon test matrix.
- Linux/Windows cloud-only artifact tests.
- SBOM, license inventory, provenance, and release attestations.
- Clean-machine install tests from packed/published artifacts.

Exit criteria:

- Installation never invokes Rust, CMake, Xcode, or `node-gyp` on a supported target.
- Unsupported targets fail with a deliberate capability message, not npm platform-resolution noise.
- A release can be reproduced from a tag.

## Phase 7 — Product migration

**Purpose:** make the Rust implementation Cuttledoc rather than a permanent side project.

Deliverables:

- Final compatibility report and benchmark comparison.
- Cuttledoc 2 → 3 migration guide.
- Cache migration or reuse behavior.
- Deprecation notices for `parakeet-coreml` and `whisper-coreml`.
- Plan for `@cuttledoc/ffmpeg` and `@cuttledoc/llm`.
- Import or merge strategy into `sebastian-software/cuttledoc`.
- Release candidate, rollback plan, and support window.

Exit criteria:

- Cuttledoc v3 is released from the product repository.
- Legacy repositories clearly point to their replacements.
- No model or package is archived before users have a working migration path.

## Cross-cutting workstreams

### Testing

- Unit tests for pure Rust transformations and state machines.
- Contract tests shared across backends.
- Real-model integration tests on self-hosted Apple Silicon runners.
- FLEURS-based regression fixtures already used by Cuttledoc.
- Packed npm tests, installed CLI tests, and crate consumer tests.
- Failure injection for partial downloads, corrupted caches, cancellations, and unavailable hardware.

### Performance

Track separately:

- decode time;
- model download and validation time;
- cold and warm model load time;
- inference real-time factor;
- peak/resident memory;
- first-result latency;
- native binary and npm install size.

Performance comparisons must use the same machines, fixtures, model versions, and power conditions.

### Security and supply chain

- Pin model manifests with source, revision, expected files, sizes, and digests.
- Write downloads to temporary paths and commit atomically after validation.
- Avoid arbitrary lifecycle downloads during package installation.
- Audit vendored native code and generated bindings.
- Keep API keys in environment variables or explicit in-memory options.

## Main risks

| Risk                                                                             | Impact                               | First mitigation                                        |
| -------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------- |
| CoreML ownership or thread affinity is incompatible with generic async executors | Crashes or nondeterministic behavior | Phase 0 single-thread executor spike                    |
| Direct Rust CoreML coverage is incomplete                                        | Large unsafe surface or stalled port | Stage through a narrow C ABI and replace incrementally  |
| whisper.cpp build and linking dominate complexity                                | Fragile releases                     | Isolated sys crate, pinned upstream, artifact test      |
| npm platform packaging recreates current install failures                        | Users cannot install                 | Pack/install matrix before backend porting expands      |
| Model downloads are huge and currently weakly verified                           | Corruption and poor UX               | Versioned manifests, hashes, atomic/resumable downloads |
| “One package” grows too large through embedded LLM inference                     | Slow installs and maintenance burden | Decide enhancement separately in Phase 5                |
| Rewrite changes recognition quality unnoticed                                    | Product regression                   | Golden fixtures and benchmark gates from Phase 1        |
| Incubator and production diverge                                                 | Duplicate maintenance                | Time-box phases and migrate into product repo at parity |

## Immediate next actions

1. Review and accept ADR-0001 through ADR-0003.
2. Select the minimal Parakeet model invocation for the CoreML interop spike.
3. Scaffold only the crates needed by that spike.
4. Add a macOS Apple Silicon CI or self-hosted test path.
5. Produce the first packed npm addon and test ESM/CommonJS loading.
