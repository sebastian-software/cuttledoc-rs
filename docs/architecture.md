# Target architecture

## Context

The current system distributes product behavior across a TypeScript monorepo and two standalone Node native addons. That separation creates four overlapping lifecycle systems:

- Node/TypeScript orchestration and backend selection in Cuttledoc;
- Objective-C++ Parakeet inference and VAD behind N-API;
- C++/whisper.cpp inference behind a second N-API addon;
- separate package installation, model caches, CLIs, tests, and releases.

The target makes Rust the product center while preserving proven native inference implementations where rewriting them would add risk without user value. Runtime and model selection remains evidence-driven and must not shape the stable API.

## System view

```text
                         ┌─────────────────────────────┐
                         │ User applications           │
                         │ Rust │ CLI │ Node/TypeScript│
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │ Public Cuttledoc API         │
                         │ engines, models, progress,   │
                         │ cancellation, stable errors │
                         └──────────────┬──────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
    ┌─────────▼─────────┐     ┌─────────▼─────────┐     ┌─────────▼─────────┐
    │ Audio pipeline     │     │ Task engines      │     │ Model manager     │
    │ probe/decode/PCM   │     │ STT │ text gen    │     │ manifest/download │
    │ normalize          │     │ lifecycle/results │     │ validate/migrate  │
    └─────────┬─────────┘     └─────────┬─────────┘     └───────────────────┘
              │                         │
              │          ┌──────────────┼──────────────┐
              │          │              │              │
              │  ┌───────▼───────┐ ┌────▼─────┐ ┌──────▼──────┐
              │  │ Apple-local    │ │ Native   │ │ Remote      │
              │  │ CoreML/System  │ │ MLX/Metal│ │ HTTP APIs   │
              │  └───────────────┘ └──────────┘ └─────────────┘
              │
              └── managed FFmpeg compatibility path initially
```

## Dependency direction

Dependencies point inward toward domain types and contracts:

```text
cuttledoc-cli ─────┐
cuttledoc-node ────┼──▶ cuttledoc ───▶ backend/audio/model abstractions
                   │                         ▲
                   └─────────────────────────┘

cuttledoc-apple/runtime adapters ─▶ abstractions
cuttledoc-openai ─────▶ abstractions
cuttledoc-audio ──────▶ domain audio types
cuttledoc-models ─────▶ model manifests and storage contracts
```

The public `cuttledoc` crate must not depend on Node.js. `cuttledoc-node` depends on the Rust API and contains only conversion, callback bridging, and Node-specific lifecycle handling.

## Task scope

The stable initial product architecture covers two composable AI tasks:

1. Speech-to-text through `SpeechRecognitionEngine`-shaped APIs.
2. Text generation through `TextGenerationEngine`-shaped APIs.

Speech synthesis remains an explicit strategic direction for a complete STT → text generation → TTS flow, but it is not part of the initial release gate. Phase 5 must validate it with a narrow vertical slice before fixing a public `SpeechSynthesisEngine`-shaped contract. It remains a separate task rather than an extension of recognition or generation (ADR-0009). Transcript enhancement composes the two initial tasks.

Embeddings, vision, image generation, and general tensor execution are outside the initial scope. Runtime adapters may be internally extensible, but no public abstraction is added speculatively for those tasks.

## Proposed crates

### `cuttledoc`

Owns:

- stable public Rust types;
- backend selection and capability reporting;
- engine construction and reuse;
- transcription orchestration;
- progress, cancellation, and cleanup semantics;
- stable error categories and codes.

It does not own platform FFI or JavaScript types.

### Apple runtime adapters (exact crate split deferred)

Own selected Apple-local inference behind task contracts:

- CoreML, Apple system Speech, and any accepted MLX/Metal/native paths;
- selected ASR model preprocessing, decoding, segmentation, and alignment;
- runtime-specific compute-unit/device configuration;
- Apple-specific thread affinity, autorelease behavior, and deterministic cleanup;
- conversion of native results into shared domain results.

Phase 0 decides whether these adapters live in one `cuttledoc-apple` crate or smaller runtime-specific crates. No crate is created merely to mirror an old npm package or an experimental dependency.

Platform code is gated to `target_os = "macos"` and `target_arch = "aarch64"` with a macOS 26 deployment baseline (ADR-0007). Other targets expose capability absence at the orchestration layer rather than compiling Apple frameworks.

Apple SpeechAnalyzer/SpeechTranscriber has no Objective-C or C interface. It is reached through a repository-owned Swift shim exposing a narrow C ABI to Rust, including AssetInventory model installation for CLI and library consumers.

### `cuttledoc-models`

Owns a versioned manifest per model artifact:

- logical model ID and backend;
- upstream URL and immutable revision when available;
- expected file tree, size, and digest;
- cache layout and compatibility version;
- partial-download directory;
- validation and atomic commit;
- progress and cancellation;
- migration from existing cache locations.

Compatibility baseline model set (the bakeoff decides what ships initially):

- Parakeet Preprocessor, Encoder, Decoder, JointDecision, vocabulary;
- Silero VAD `silero-vad-unified-v6.0.0.mlmodelc`;
- Whisper `ggml-large-v3-turbo.bin`;
- Whisper `ggml-large-v3-turbo-encoder.mlmodelc`.

### `cuttledoc-audio`

Owns:

- media probing;
- decode to interleaved PCM;
- conversion to mono 16 kHz `f32`;
- duration calculation;
- normalization policy;
- temporary-file lifecycle;
- configured or managed FFmpeg resolution.

The compatibility-first implementation may continue invoking FFmpeg. Replacing selected audio-only formats with Rust decoders is an optimization, not a prerequisite.

### `cuttledoc-openai`

Owns cloud transcription requests and response conversion. It receives credentials from in-memory configuration and must never persist or log them.

### `cuttledoc-cli`

Owns command parsing and terminal presentation only. Commands call the same Rust API used by bindings.

### `cuttledoc-node`

Owns:

- `napi-rs` annotations and generated TypeScript declarations;
- conversion between JS objects, buffers/typed arrays, and Rust types;
- Promise, progress callback, and cancellation bridging;
- Node-specific object finalizers.

It must not select backends, manage model paths independently, or implement transcription behavior.

## Runtime model

### Engine lifecycle

Engines are explicit, reusable resources:

```text
Configured → Loading → Ready → Running → Ready → Closing → Closed
                 └──────────── failure ───────────────▶ Failed
```

Rules:

- Initialization is idempotent.
- One engine does not execute concurrent transcriptions unless the backend proves it safe.
- The public API queues, rejects, or creates multiple engines deliberately; it never races a global native singleton accidentally.
- Cleanup is explicit and also guarded by `Drop`/Node finalizers.
- Errors preserve backend and phase context.

### Apple-local execution

Phase 0 must determine thread affinity, run-loop, executor, and cleanup rules separately for each candidate runtime. The CoreML adapter must preserve concurrent async prediction where supported, bound in-flight work to control peak memory, serialize synchronous prediction on a model instance, and serialize predictions that share one `MLState`. All Objective-C objects must live inside bounded autorelease pools. MLX or Metal behavior must be measured rather than assumed equivalent.

### Progress and cancellation

Long operations emit typed events:

- resolving model;
- downloading bytes/files;
- validating artifacts;
- decoding media;
- loading engine;
- transcribing segment;
- generating text;
- enhancing transcript.

Speech-synthesis progress events are added only after the Phase 5 vertical slice validates their meaning (ADR-0009).

Cancellation is cooperative. Downloads remove or retain resumable partial state according to the model-manager policy. Native inference that cannot be interrupted reports cancellation at the next safe boundary.

Transcription results are delivered as an ordered stream of range-addressed replace/revoke updates per ADR-0008. Replacement content is volatile or final; finalized ranges are immutable. Progress events remain separate and observational.

## Apple runtime and native interop

Rust remains the owner even when foreign implementations are reused:

- Existing Parakeet Objective-C++ and Whisper C++ implementations are compatibility references and may be temporary bridges only when that is the smallest maintainable path.
- The primary MLX spike uses the official C++ core through the smallest repository-owned task-level adapter; official `mlx-c` is the comparison path. Community bindings remain reference-only unless they pass ADR-0005.
- Specialized upstream implementations such as whisper.cpp remain candidates rather than automatic dependencies.
- Unsafe code and foreign pointers remain isolated in sys/interop modules.
- Public Rust types never expose Objective-C, C++, N-API, CoreML, MLX, Metal, or whisper.cpp handles.
- Ownership, nullability, thread affinity, error transfer, and cleanup are documented at every FFI boundary.

The Phase 0 bakeoff selects model/runtime pairs and then records each production interop boundary in a follow-up ADR. See ADR-0003, ADR-0005, and ADR-0006.

## Dependency acceptance

Every material runtime, wrapper, build component, and code generator is classified as an accepted production dependency, repository-owned boundary, reference-only input, or rejected project. A working spike does not bypass this review. Experimental projects may inform tests and implementation without entering Cargo/npm manifests or release artifacts. See ADR-0005.

## Unified model storage

Proposed new root:

```text
~/.cache/cuttledoc/
├── models/
│   ├── parakeet-tdt-0.6b-v3/
│   ├── silero-vad-v6/
│   └── whisper-large-v3-turbo/
├── manifests/
└── downloads/
```

Existing roots:

- `~/.cache/parakeet-coreml/models`
- `~/.cache/parakeet-coreml/vad`
- `~/.cache/whisper-coreml/models`

The model manager should detect complete existing caches and either reuse them read-only, adopt them after verification, or provide an explicit migration. It must not redownload several gigabytes without explanation.

The `~/.cache` root is a deliberate decision: it matches the existing Parakeet/Whisper cache convention and keeps one path scheme across macOS and future cloud-only targets, instead of the macOS-native `~/Library/Caches`. System Speech models are managed by macOS through AssetInventory and never live in this cache.

## Error model

Internal errors remain rich Rust enums. Public boundaries expose stable categories and codes, for example:

- `INVALID_INPUT`
- `UNSUPPORTED_PLATFORM`
- `BACKEND_UNAVAILABLE`
- `MODEL_MISSING`
- `MODEL_INVALID`
- `MODEL_DOWNLOAD_FAILED`
- `AUDIO_DECODE_FAILED`
- `ENGINE_LOAD_FAILED`
- `TRANSCRIPTION_FAILED`
- `AUTHENTICATION_FAILED`
- `CANCELLED`

Node errors include `code`, backend/model context, and the native cause when safe. CLI errors map the same codes to actionable messages and nonzero exit statuses.

## Distribution architecture

The user-facing npm contract is one package named `cuttledoc`. Implementation-only packages such as `@cuttledoc/darwin-arm64` may carry prebuilt artifacts if npm distribution requires them, but they expose no product API and are versioned automatically with the facade.

Every release tests:

1. Rust workspace from source.
2. Final native CLI archive.
3. Final npm tarball installed in an empty project.
4. ESM import.
5. CommonJS require.
6. Native addon load.
7. At least one real transcription on supported Apple Silicon infrastructure.

## Observability

The library does not log by default. It emits structured progress and diagnostic events that callers may render. Sensitive paths, credentials, and transcript content require explicit opt-in before appearing in diagnostics.

Benchmark reports record:

- Cuttledoc and model versions;
- hardware and OS;
- backend configuration;
- input fixture and duration;
- cold/warm state;
- timing and memory measurements;
- recognition metrics such as WER.
