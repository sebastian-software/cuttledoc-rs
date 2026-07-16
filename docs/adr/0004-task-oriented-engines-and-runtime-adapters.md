# ADR-0004: Task-oriented engines and internal runtime adapters

## Status

Accepted

## Context

CoreML, MLX, Metal-native frameworks, whisper.cpp, Apple Speech APIs, and remote APIs expose materially different execution models. A generic CoreML-to-Rust or universal tensor abstraction would either leak runtime details into Cuttledoc or collapse useful capabilities into a lowest common denominator.

Cuttledoc needs stable product concepts: transcribing speech, optionally generating text, managing models, reporting capabilities, cancellation, progress, and deterministic resource ownership.

## Decision

Define stable APIs around tasks and domain results rather than inference runtimes.

- Speech recognition is exposed through a `SpeechEngine`-shaped contract.
- Optional transcript enhancement or local generation uses a separate `TextGenerationEngine`-shaped contract.
- Future tasks such as embeddings or audio classification receive separate contracts only when required.
- Runtime adapters remain internal and may use CoreML, MLX, Metal, C/C++, Apple system APIs, or remote HTTP independently.
- Model identity and runtime identity remain distinct. Multiple runtimes may execute the same model family without changing transcription result types.
- Shared infrastructure is limited to genuine common concerns: model identity and state, installation and verification, capabilities, progress, cancellation, diagnostics, errors, and lifecycle.
- Stable APIs never expose raw tensors, Objective-C objects, C++ contexts, MLX arrays, CoreML models, or Node handles.

The public API may support an advanced runtime preference later, but it is a capability-gated hint rather than the organizing abstraction.

## Consequences

### Positive

- Runtime and model choices can evolve without redesigning the product API.
- Speech and LLM decisions can be made independently.
- Platform-specific ownership stays behind narrow boundaries.
- Node remains a mechanical binding over the Rust task API.

### Negative

- Some lifecycle and option types cannot be shared across tasks.
- Runtime-specific diagnostics require explicit capability structures.
- Internal adapters may look different instead of satisfying one convenient universal trait.

## Alternatives considered

### Public generic tensor/runtime API

This would turn Cuttledoc into a machine-learning framework and couple callers to implementation details that do not help the transcription product.

### One engine trait for ASR and LLMs

Speech recognition and autoregressive text generation have different inputs, outputs, streaming semantics, and model state. Their limited shared lifecycle does not justify one execution contract.

### Runtime-specific public packages

This recreates the fragmentation ADR-0001 is intended to remove.

## Validation

Phase 0 API sketches and spikes must show that at least two distinct local runtimes can map into the speech contract without exposing their handles. The LLM evaluation must use a separate generation contract.
