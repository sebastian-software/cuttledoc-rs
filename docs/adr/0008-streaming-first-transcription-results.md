# ADR-0008: Streaming-first transcription results

## Status

Accepted

## Context

Cuttledoc 2 and the current native packages are batch-only: the caller receives one final result. Several relevant backends are streaming-first or can emit results incrementally:

- Apple SpeechTranscriber emits volatile (revisable) results followed by finalized results.
- Parakeet with VAD segmentation and chunked Whisper decoding can emit finalized segments as they complete.
- Remote APIs vary by model.

Retrofitting a streaming result model onto a batch-only stable API would be a breaking redesign. Deciding this before the first vertical slice is cheaper than after.

## Decision

Streaming transcription is part of the v3 scope.

- The core result contract is an ordered stream of typed, range-addressed updates. Each update has a monotonically increasing sequence number and an affected audio time range.
- An update either **replaces** the current non-final content in its affected range with volatile or final segments, or **revokes** the current non-final content in that range. This represents backends that revise segmentation as well as Apple SpeechTranscriber results that explicitly revoke earlier volatile text.
- Final content is immutable. An update that overlaps an already finalized range is a backend contract violation rather than an implicit revision.
- Batch transcription is the degenerate case of the same contract: only final updates, aggregated into the existing `TranscriptionResult`. One-shot APIs keep returning the aggregated final result.
- Streaming behavior is capability-reported per backend: `supports_streaming` (emits finalized results incrementally) and `emits_volatile_results` (may revise before finalizing). Callers must be able to rely on the capability report rather than probing behavior.
- v3 input sources are files and caller-provided PCM feeds. Microphone or system-audio capture is out of scope for the library; callers own capture and the associated OS permissions.
- Rust exposes an async stream of updates; Node exposes an `AsyncIterator`; the CLI may render progressive output. Progress events (ADR/architecture progress model) remain separate from result updates.

## Consequences

### Positive

- The API does not need a breaking redesign when streaming-capable backends (system Speech, future realtime models) ship.
- Batch-only backends integrate by emitting final segments as they complete — no artificial dual code path.
- Plays to the strengths of the macOS 26 baseline (ADR-0007) and streaming-first candidates in the bakeoff.

### Negative

- The engine contract and bindings are more complex than a single promise/future.
- Backends must map their native revision model into sequence/range/replace/revoke semantics; contract tests must cover replacement, revocation, overlap, and volatile-to-final sequences.
- The compatibility gates gain a streaming dimension.

## Alternatives considered

### Batch-only v3 with an API reserve

Cheapest now, but the reserve is hard to guarantee: result types, engine methods, and Node bindings all shape around a single final value, and history shows such reserves rarely survive contact with implementation.

### Streaming including library-owned audio capture

Pulls microphone/screen-audio permissions, device handling, and TCC attribution into the library. Capture stays with the caller.

## Validation

Issue #8 (engine/backend boundaries) must incorporate this contract. Phase 0 must map at least one volatile-capable backend (system Speech) and one finals-only backend (existing CoreML path) into the same update stream without leaking runtime handles. A reducer shared by Rust/Node contract tests must deterministically reconstruct the same final transcript from replace, revoke, and volatile-to-final sequences.
