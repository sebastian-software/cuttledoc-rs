# ADR-0003: Staged native interop instead of an all-at-once native rewrite

## Status

Proposed

## Context

The current local backends contain proven native implementations:

- Parakeet uses Objective-C++ to drive CoreML models, a transducer decoder, mel preprocessing, and stateful Silero VAD.
- Whisper uses C++ and a pinned whisper.cpp build with CoreML, Metal, Accelerate, and Apple frameworks.

Rust can own product behavior without reimplementing mature inference runtimes immediately. An all-at-once rewrite would combine language migration, FFI migration, recognition parity, model lifecycle changes, and package changes into one untestable risk.

Direct Rust CoreML bindings may eventually reduce foreign code, but their API coverage, ownership ergonomics, and thread behavior must be demonstrated against the models used here.

## Decision

Use a staged interop strategy:

1. Rust owns engine lifecycle, public types, orchestration, model management, errors, and concurrency from the first vertical slice.
2. Isolate existing native inference behind the narrowest practical C ABI or sys crate.
3. Keep whisper.cpp as an upstream C/C++ inference dependency rather than rewriting Whisper.
4. Evaluate direct Rust CoreML bindings during Phase 0 using a real Parakeet model invocation.
5. Port self-contained algorithms such as vocabulary handling, result merging, download logic, and potentially the transducer decoder to safe Rust when golden tests establish parity.
6. Keep all unsafe/foreign ownership inside dedicated interop modules.

No Node-API code is retained below the Rust product API. `napi-rs` exists only at the outer Node adapter.

## Required FFI contract

Every foreign boundary documents:

- allocator and owner for every pointer/buffer/string;
- nullability and error transfer;
- thread creation and affinity;
- autorelease-pool requirements;
- whether calls block;
- reentrancy and concurrency;
- cleanup after partial initialization;
- panic/exception boundaries;
- deployment target and linked frameworks.

C++ exceptions, Objective-C exceptions, and Rust panics must never cross the ABI.

## Consequences

### Positive

- Fastest route to measurable recognition parity.
- Rust becomes the architecture center without pretending foreign runtimes disappear.
- Native migration can proceed component by component.
- Existing models and performance characteristics remain testable.
- The Node binding is simplified immediately.

### Negative

- The first implementation is not “pure Rust.”
- CMake/Xcode/Cargo integration remains necessary in release CI.
- A temporary C/Objective-C++ bridge may later be replaced.
- Unsafe ownership errors remain possible and demand strong tests.

## Alternatives considered

### Rewrite Parakeet directly with Rust CoreML APIs before any vertical slice

Potentially cleaner long term, but it front-loads uncertainty about framework coverage, Objective-C runtime ownership, multi-array conversion, model state, and thread affinity.

### Keep the current Node addons and call them from Rust through Node

This leaves Node as the real runtime and defeats the Rust library/CLI objective.

### Rewrite whisper.cpp in Rust

This duplicates an active specialized upstream project and offers little product value relative to the risk.

## Validation

Phase 0 must compare at least two Parakeet paths:

- direct Rust CoreML framework calls;
- Rust calling a narrow bridge around the existing Objective-C++ engine.

Accept the simplest path that demonstrates correct ownership, real inference, deterministic cleanup, acceptable binary size, and a maintainable build. Record the selected bridge in a follow-up ADR before Phase 2 expands it.
