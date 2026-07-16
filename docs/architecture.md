# Target architecture

## Context

The current system distributes product behavior across a TypeScript monorepo and two standalone Node native addons. That separation creates four overlapping lifecycle systems:

- Node/TypeScript orchestration and backend selection in Cuttledoc;
- Objective-C++ Parakeet inference and VAD behind N-API;
- C++/whisper.cpp inference behind a second N-API addon;
- separate package installation, model caches, CLIs, tests, and releases.

The target makes Rust the product center while preserving proven native inference implementations where rewriting them would add risk without user value.

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
    │ Audio pipeline     │     │ Backend runtime   │     │ Model manager     │
    │ probe/decode/PCM   │     │ select/load/cache │     │ manifest/download │
    │ normalize          │     │ transcribe/dispose│     │ validate/migrate  │
    └─────────┬─────────┘     └─────────┬─────────┘     └───────────────────┘
              │                         │
              │          ┌──────────────┼──────────────┐
              │          │              │              │
              │  ┌───────▼───────┐ ┌────▼─────┐ ┌──────▼──────┐
              │  │ Parakeet       │ │ Whisper  │ │ OpenAI      │
              │  │ CoreML + VAD   │ │ cpp/CoreML│ │ HTTP API    │
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

cuttledoc-coreml ─────▶ abstractions
cuttledoc-openai ─────▶ abstractions
cuttledoc-audio ──────▶ domain audio types
cuttledoc-models ─────▶ model manifests and storage contracts
```

The public `cuttledoc` crate must not depend on Node.js. `cuttledoc-node` depends on the Rust API and contains only conversion, callback bridging, and Node-specific lifecycle handling.

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

### `cuttledoc-coreml`

Owns Apple-local inference:

- Parakeet CoreML model lifecycle;
- Silero VAD state and segmentation;
- 15-second Parakeet chunk enforcement;
- Whisper/whisper.cpp lifecycle with CoreML encoder;
- CoreML compute-unit configuration;
- Apple-specific thread-affinity and autorelease behavior;
- conversion of native results into shared domain results.

Platform code is gated to `target_os = "macos"` and `target_arch = "aarch64"`. Other targets expose capability absence at the orchestration layer rather than compiling Apple frameworks.

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

Initial model set:

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
Configured → Loading → Ready → Transcribing → Ready → Closing → Closed
                 └──────────── failure ───────────────▶ Failed
```

Rules:

- Initialization is idempotent.
- One engine does not execute concurrent transcriptions unless the backend proves it safe.
- The public API queues, rejects, or creates multiple engines deliberately; it never races a global native singleton accidentally.
- Cleanup is explicit and also guarded by `Drop`/Node finalizers.
- Errors preserve backend and phase context.

### CoreML execution

Phase 0 must determine whether CoreML objects need a dedicated thread/run loop. Until proven otherwise, treat each engine as thread-affine and send commands to a dedicated worker. All Objective-C objects must live inside bounded autorelease pools.

### Progress and cancellation

Long operations emit typed events:

- resolving model;
- downloading bytes/files;
- validating artifacts;
- decoding media;
- loading engine;
- transcribing segment;
- enhancing transcript.

Cancellation is cooperative. Downloads remove or retain resumable partial state according to the model-manager policy. Native inference that cannot be interrupted reports cancellation at the next safe boundary.

## CoreML and native interop

Rust remains the owner even when foreign implementations are reused:

- Parakeet may initially expose the current Objective-C++ engine through a narrow C ABI.
- Whisper continues to use whisper.cpp rather than reimplementing Whisper inference.
- Unsafe code and foreign pointers remain isolated in sys/interop modules.
- Public Rust types never expose Objective-C, C++, N-API, or whisper.cpp handles.
- Ownership, nullability, thread affinity, error transfer, and cleanup are documented at every FFI boundary.

The Phase 0 spike decides whether Parakeet moves directly to Rust CoreML bindings or stages through the C ABI. See ADR-0003.

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

## Error model

Internal errors remain rich Rust enums. Public boundaries expose stable categories and codes, for example:

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
