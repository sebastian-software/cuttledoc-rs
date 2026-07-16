# ADR-0004: Task-oriented engines and internal runtime adapters

## Status

Accepted

## Context

CoreML, MLX, Metal-native frameworks, whisper.cpp, Apple Speech APIs, and remote APIs expose materially different execution models. A generic CoreML-to-Rust or universal tensor abstraction would either leak runtime details into Cuttledoc or collapse useful capabilities into a lowest common denominator.

Cuttledoc needs stable product concepts for three initial AI tasks: speech recognition, speech synthesis, and text generation. Together they support transcription, transcript enhancement, and complete speech → text → LLM → speech pipelines. They also need shared model management, capabilities, cancellation, progress, streaming, backpressure, and deterministic resource ownership.

## Decision

Define stable APIs around tasks and domain results rather than inference runtimes.

- Speech-to-text is exposed through a `SpeechRecognitionEngine`-shaped contract.
- Text-to-speech is exposed through a separate `SpeechSynthesisEngine`-shaped contract.
- Transcript enhancement and local text generation use a `TextGenerationEngine`-shaped contract.
- Speech recognition and synthesis share audio formats, PCM buffers, streaming, and lifecycle vocabulary, but not one execution trait.
- Streaming is part of all three contracts: partial/final transcript events, generated text deltas, and synthesized audio chunks.
- Embeddings, vision, image generation, and arbitrary tensor execution are out of the initial architecture scope. They receive separate contracts only after a real product requirement exists.
- Runtime adapters remain internal and may use CoreML, MLX, Metal, C/C++, Apple system APIs, or remote HTTP independently.
- Model identity and runtime identity remain distinct. Multiple runtimes may execute the same model family without changing transcription result types.
- Shared infrastructure is limited to genuine common concerns: model identity and state, installation and verification, capabilities, progress, cancellation, backpressure, diagnostics, errors, lifecycle, and common audio buffer types where applicable.
- Stable APIs never expose raw tensors, Objective-C objects, C++ contexts, MLX arrays, CoreML models, or Node handles.

The public API may support an advanced runtime preference later, but it is a capability-gated hint rather than the organizing abstraction.

## Consequences

### Positive

- Runtime and model choices can evolve without redesigning the product API.
- STT, TTS, and LLM decisions can be made independently and composed into a voice pipeline.
- Platform-specific ownership stays behind narrow boundaries.
- Node remains a mechanical binding over the Rust task API.

### Negative

- Some lifecycle and option types cannot be shared across tasks.
- Runtime-specific diagnostics require explicit capability structures.
- Internal adapters may look different instead of satisfying one convenient universal trait.

## Alternatives considered

### Public generic tensor/runtime API

This would turn Cuttledoc into a machine-learning framework and couple callers to implementation details that do not help the transcription product.

### One engine trait for STT, TTS, and LLMs

Speech recognition, speech synthesis, and autoregressive text generation have different inputs, outputs, streaming directions, options, and model state. Their shared lifecycle does not justify one execution contract.

### Runtime-specific public packages

This recreates the fragmentation ADR-0001 is intended to remove.

## Validation

Phase 0 API sketches must define all three contracts and their composition without exposing runtime handles. Runtime spikes must show that distinct local runtimes can map into the speech-recognition contract. LLM evaluation uses the separate generation contract. TTS runtime/model selection may be deferred beyond the first ASR release, but the audio output and synthesis streaming contract must not be retrofitted through the recognition API.
