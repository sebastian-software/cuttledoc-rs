# ADR-0003: Staged, runtime-neutral native interop

## Status

Accepted

## Context

The current local backends contain proven native implementations:

- Parakeet uses Objective-C++ to drive CoreML models, a transducer decoder, mel preprocessing, and stateful Silero VAD.
- Whisper uses C++ and a pinned whisper.cpp build with CoreML, Metal, Accelerate, and Apple frameworks.

Rust can own product behavior without reimplementing mature inference runtimes immediately. An all-at-once rewrite would combine language migration, FFI migration, recognition parity, model lifecycle changes, and package changes into one untestable risk.

Direct Rust CoreML bindings may reduce foreign code, but their API coverage, ownership ergonomics, dependency quality, and thread behavior must be demonstrated against the models used here. MLX, Metal-native runtimes, and Apple system Speech APIs are additional candidates with different interop and maintenance profiles.

The existence of prior Objective-C++, C++, or Node implementations is evidence and reference material. It does not make their package boundary or binding technology the default target architecture.

## Decision

Use a staged interop strategy:

1. Rust owns engine lifecycle, public types, orchestration, model management, errors, and concurrency from the first vertical slice.
2. Put each selected runtime behind a task-specific internal adapter. Do not expose a generic tensor runtime or foreign handles through the product API.
3. Prefer stable platform APIs and established upstream libraries. When no acceptable Rust binding exists, evaluate the smallest repository-owned C ABI or sys boundary over the established upstream API.
4. Treat existing Objective-C++ and C++ engines as parity references and possible temporary bridges, not mandatory foundations.
5. Keep maintained specialized upstream implementations such as whisper.cpp where they pass the dependency policy and replacing them offers no product value.
6. MLX integration may ship only through a dependency that passes ADR-0005 or through a deliberately small repository-owned boundary whose maintenance is explicitly accepted.
7. Port self-contained algorithms such as vocabulary handling, result merging, download logic, and potentially transducer decoding to safe Rust when golden tests establish parity.
8. Keep all unsafe/foreign ownership inside dedicated interop modules.

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

- Separates product architecture from the winner of the runtime/model bakeoff.
- Fast route to measurable recognition parity without committing to an immature wrapper.
- Rust becomes the architecture center without pretending foreign runtimes disappear.
- Native migration can proceed component by component.
- Existing models and performance characteristics remain testable.
- The Node binding is simplified immediately.

### Negative

- The first implementation is not “pure Rust.”
- CMake/Xcode/Cargo integration remains necessary in release CI.
- Temporary C, Objective-C++, or C++ bridges may later be replaced.
- Unsafe ownership errors remain possible and demand strong tests.

## Alternatives considered

### Choose CoreML, MLX, or an existing bridge before measuring them

This would turn an implementation hypothesis into an architectural dependency before runtime quality, ownership, packaging, and maintenance are known.

### Keep the current Node addons and call them from Rust through Node

This leaves Node as the real runtime and defeats the Rust library/CLI objective.

### Rewrite whisper.cpp in Rust

This duplicates an active specialized upstream project and offers little product value relative to the risk.

## Validation

Phase 0 must exercise representative CoreML and MLX paths from Rust and measure the Apple system Speech path. Existing Parakeet and Whisper implementations remain baselines. Accept only paths that demonstrate correct ownership, real inference, deterministic cleanup, acceptable distribution cost, and compliance with ADR-0005. Record each selected production boundary in a follow-up ADR before expanding a vertical slice.
