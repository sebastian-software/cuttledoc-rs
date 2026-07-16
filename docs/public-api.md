# Proposed public APIs

## Status

This document is a design target, not a compatibility promise. Names remain provisional until the first vertical slice proves the ownership and async model.

## API layers

Cuttledoc exposes two deliberate levels:

1. A high-level one-shot API for most applications.
2. A reusable engine API for loaded models, repeated transcription, explicit lifecycle, progress, and cancellation.

Model management and capability discovery are first-class. Backend implementation objects are not exposed.

Speech recognition, speech synthesis, and text generation are separate task APIs. They may share model management, progress, cancellation, diagnostics, errors, lifecycle concepts, and common audio types where applicable, but they do not share a generic tensor or inference-runtime interface. See ADR-0004.

The `Backend` identifiers below remain provisional compatibility/product choices, not runtime identities. CoreML, MLX, Metal, Apple system APIs, or a native upstream implementation remain internal adapters. If multiple runtimes can execute the same model family, runtime selection does not change `TranscriptionResult`.

## Initial task contracts

The following shapes establish architectural boundaries; exact Rust async syntax and names remain provisional:

```rust
pub trait SpeechRecognitionEngine {
    async fn transcribe(
        &self,
        audio: AudioSource,
        options: TranscriptionOptions,
    ) -> Result<TranscriptionStream>;
}

pub trait SpeechSynthesisEngine {
    async fn synthesize(
        &self,
        text: TextSource,
        options: SynthesisOptions,
    ) -> Result<AudioStream>;
}

pub trait TextGenerationEngine {
    async fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<TextStream>;
}
```

Shared audio primitives support file, in-memory PCM, and streaming use cases in both directions:

```rust
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_format: SampleFormat,
}

pub struct AudioChunk {
    pub samples: AudioSamples,
    pub format: AudioFormat,
    pub timestamp: Option<Duration>,
}
```

Recognition streams partial/final transcript events, generation streams text/token deltas, and synthesis streams `AudioChunk` values. All streams define cancellation and backpressure behavior. TTS model delivery may follow the first ASR release, but the synthesis contract is part of the architecture now.

## Shared domain model

```rust
#[non_exhaustive]
pub enum Backend {
    Auto,
    Parakeet,
    Whisper,
    AppleSpeech, // Apple SpeechAnalyzer, macOS 26+ (ADR-0007)
    OpenAi,
}

pub struct TranscribeOptions {
    pub backend: Backend,
    pub language: Option<String>,
    pub openai_api_key: Option<SecretString>,
    pub openai_model: Option<String>,
}

pub struct TranscriptionResult {
    pub text: String,
    pub language: Option<String>,
    pub backend: Backend,
    pub media_duration: Duration,
    pub processing_duration: Duration,
    pub segments: Vec<TranscriptionSegment>,
}

pub struct TranscriptionSegment {
    pub text: String,
    pub start: Duration,
    pub end: Duration,
    pub confidence: Option<f32>,
    pub words: Vec<WordTimestamp>,
}
```

`Backend` is `#[non_exhaustive]`: downstream Rust `match` expressions must carry a wildcard arm, which permits adding variants after 1.0. TypeScript has no equivalent closed-enum guarantee: a plain string union can break exhaustive consumer switches when widened. The Node surface therefore uses known completions plus an open string escape hatch:

```ts
type KnownBackend = "auto" | "parakeet" | "whisper" | "apple-speech" | "openai";
type BackendId = KnownBackend | (string & {});
```

The exact backend taxonomy is still provisional because the current list mixes legacy model families and providers. Issue #8 must decide the stable identifiers. Capability discovery, not an exhaustive enum or union, is the source of truth for availability.

Time is strongly typed in Rust. Node conversions use seconds consistently at the public boundary rather than mixing Parakeet seconds and Whisper milliseconds.

## Rust one-shot API

```rust
use cuttledoc::{transcribe, Backend, TranscribeOptions};

let result = transcribe(
    "meeting.mp4",
    TranscribeOptions {
        backend: Backend::Auto,
        language: Some("de".into()),
        ..Default::default()
    },
)
.await?;

println!("{}", result.text);
```

The one-shot API may use an internal bounded engine cache, but its behavior must be deterministic and cleanup must be available at process/service boundaries.

## Rust reusable engine API

```rust
use cuttledoc::{Backend, Engine, EngineOptions};

let engine = Engine::builder()
    .backend(Backend::Parakeet)
    .language("de")
    .model_root("/srv/cuttledoc/models")
    .auto_download(false)
    .build()
    .await?;

let result = engine.transcribe_path("meeting.mp4").await?;

// Advanced callers can bypass media decoding.
let result = engine
    .transcribe_pcm(&samples, 16_000, 1)
    .await?;

engine.close().await?;
```

Open questions for the spike:

- Whether `Engine` can be `Send`/`Sync` or is a handle to a thread-affine worker.
- Whether concurrent calls queue or return a busy error.
- Whether the one-shot cache is process-global or owned by a `Cuttledoc` context.

The preferred service-oriented shape is an explicit context:

```rust
let cuttledoc = Cuttledoc::builder()
    .cache_root("/srv/cuttledoc")
    .build()?;

let engine = cuttledoc.engine(Backend::Whisper).build().await?;
```

## Model API

```rust
let models = cuttledoc.models();

for model in models.list().await? {
    println!("{} {:?}", model.id, model.state);
}

models
    .download("parakeet-tdt-0.6b-v3")
    .progress(|event| println!("{event:?}"))
    .await?;

models.verify("whisper-large-v3-turbo").await?;
models.remove("whisper-large-v3-turbo").await?;
```

Proposed model states:

- `missing`
- `partial`
- `downloading`
- `verifying`
- `ready`
- `invalid`

Model IDs are stable product identifiers. Upstream filenames remain manifest details.

## Capabilities API

```rust
let capabilities = cuttledoc.capabilities().await;

for backend in capabilities.backends {
    println!(
        "{:?}: available={}, languages={}",
        backend.id,
        backend.available,
        backend.languages.len()
    );
}
```

Capability results explain unavailability:

- unsupported OS/architecture;
- missing model;
- missing API key;
- failed native load;
- unavailable managed FFmpeg binary.

## Progress and cancellation

```rust
let cancellation = CancellationToken::new();

let result = engine
    .transcribe("meeting.mp4")
    .progress(|event| match event {
        ProgressEvent::Decoding { percent } => { /* ... */ }
        ProgressEvent::Transcribing { segment, total } => { /* ... */ }
        _ => {}
    })
    .cancellation(cancellation.clone())
    .await?;
```

Progress callbacks are observational: slow callbacks must not block native inference indefinitely. Node callbacks are dispatched through a threadsafe function or equivalent `napi-rs` mechanism.

## Streaming results

Streaming follows ADR-0008: the result contract is an ordered stream of range-addressed replacement or revocation updates. Replacement content is volatile or final. Batch transcription consumes the same stream and aggregates finalized ranges.

```rust
pub struct TranscriptionUpdate {
    pub sequence: u64,
    pub affected_range: TimeRange,
    pub kind: TranscriptionUpdateKind,
}

pub enum TranscriptionUpdateKind {
    Replace {
        segments: Vec<TranscriptionSegment>,
        stability: Stability,
    },
    Revoke,
}

pub enum Stability {
    Volatile,
    Final,
}
```

```rust
let mut updates = engine.transcribe_stream("meeting.mp4").await?;

while let Some(update) = updates.next().await {
    let TranscriptionUpdate {
        affected_range,
        kind,
        ..
    } = update?;

    match kind {
        TranscriptionUpdateKind::Replace { segments, stability } => {
            transcript.replace(affected_range, segments, stability)?;
        }
        TranscriptionUpdateKind::Revoke => {
            transcript.revoke(affected_range)?;
        }
    }
}
```

```ts
for await (const update of engine.transcribeStream("meeting.mp4")) {
  transcript.apply(update); // sequence + affectedRange + replace/revoke
}
```

Rules:

- Backends report `supportsStreaming` (incremental finals) and `emitsVolatileResults` (revisions before finalization) as capabilities; callers rely on the report, not probing.
- Consumers apply updates in `sequence` order. `replace` removes prior non-final content in `affectedRange` before inserting the new segments; `revoke` removes it without replacement.
- Finalized ranges are immutable. Overlap with final content is surfaced as a backend contract error.
- Finals-only backends emit completed segments as they are produced; they never emit volatile updates.
- Input sources are files and caller-provided PCM feeds. The library does not capture microphone or system audio.
- Result updates are distinct from progress events; progress remains observational.

## Node.js API

The Node API mirrors domain concepts while remaining idiomatic TypeScript:

```ts
import { Cuttledoc, transcribe } from "cuttledoc";

const result = await transcribe("meeting.mp4", {
  backend: "auto",
  language: "de",
});

const cuttledoc = await Cuttledoc.create({
  cacheRoot: "/srv/cuttledoc",
});

const engine = await cuttledoc.createEngine({
  backend: "parakeet",
  language: "de",
  autoDownload: false,
});

const repeated = await engine.transcribe("interview.wav", {
  signal: abortController.signal,
  onProgress(event) {
    console.log(event.phase, event.percent);
  },
});

await engine.close();
await cuttledoc.close();
```

Advanced PCM input uses `Float32Array` without copying where Node-API ownership permits it:

```ts
await engine.transcribePcm(samples, {
  sampleRate: 16_000,
  channels: 1,
});
```

## Node errors

```ts
try {
  await engine.transcribe("meeting.mp4");
} catch (error) {
  if (error instanceof CuttledocError) {
    console.error(error.code, error.backend, error.model);
  }
}
```

The generated declaration should resemble:

```ts
type CuttledocErrorCode =
  | "INVALID_INPUT"
  | "UNSUPPORTED_PLATFORM"
  | "BACKEND_UNAVAILABLE"
  | "MODEL_MISSING"
  | "MODEL_INVALID"
  | "MODEL_DOWNLOAD_FAILED"
  | "AUDIO_DECODE_FAILED"
  | "ENGINE_LOAD_FAILED"
  | "TRANSCRIPTION_FAILED"
  | "AUTHENTICATION_FAILED"
  | "CANCELLED";
```

## CLI contract

Proposed commands:

```text
cuttledoc <input> [--backend auto|parakeet|whisper|openai]
                  [--language <code>]
                  [--output <path>]
                  [--format text|markdown|json]

cuttledoc models list
cuttledoc models download <model|all|asr>
cuttledoc models verify <model|all>
cuttledoc models remove <model>
cuttledoc backends list
cuttledoc doctor
cuttledoc benchmark <input> --reference <text>
```

Compatibility aliases from Cuttledoc 2 should be retained where they do not compromise the new model.

## Low-level exposure policy

Expose capabilities rather than implementation classes. The public API may offer PCM input, model paths, VAD tuning, and backend-specific option structs, but it should not expose:

- raw CoreML model handles;
- MLX arrays or runtime handles;
- Metal devices or command queues;
- N-API addon functions;
- whisper.cpp contexts;
- Objective-C pointers;
- cache-internal filenames;
- backend-global singleton state.

If an advanced capability cannot be expressed without exposing implementation details, first determine whether it belongs in a separate explicitly unstable module.

## Compatibility policy

- Rust follows semantic versioning once the crate is published.
- Node follows semantic versioning and documents Cuttledoc 2 migration differences.
- CLI output intended for humans may evolve; `--format json` receives a versioned schema.
- Model manifests are versioned independently of package releases but pinned by release metadata.
- Stable error codes outlive exact error messages.
