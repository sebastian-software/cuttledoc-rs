# ADR-0012: Use official MLX as the Voxtral runtime boundary

## Status

Accepted

## Context

Voxtral Mini 4B Realtime is a serious Apple-local ASR candidate. The pinned
Python `mlx-audio` implementation proved model quality but failed Cuttledoc's
bounded streaming and cancellation requirements. A repository-owned adapter
now implements the model graph, caches, delay conditioning, tokenizer, and
stream lifecycle directly with the official `ml-explore/mlx` C++ core.

The publisher also links `antirez/voxtral.c`, a compact pure-C inference
implementation using Metal and Metal Performance Shaders. Issue #19 required
a focused comparison before fixing the candidate's native boundary.

## Decision

Use pinned official MLX through a narrow repository-owned C++ adapter and C
ABI as the accepted Voxtral runtime boundary.

Rust owns queue bounds, per-step work limits, backpressure, cancellation,
busy/reentrancy status, explicit close, errors, and public transcript updates.
The adapter owns the Voxtral-specific model graph and state. Official MLX owns
tensor execution, memory scheduling, CPU/GPU devices, and Metal kernels.
Neither `mlx-audio` nor a prebuilt Voxtral session is a production dependency.

Keep `antirez/voxtral.c` as reference-only implementation and packaging
evidence. Do not make it a dependency and do not fork it merely to recreate
the lifecycle contract already proven over official MLX.

Use 320 ms input as the measured direct-session cadence. Retain 80 ms as a
backpressure and executor-stall stress case, not as the production default.
This ADR selects an integration boundary; it does not select Voxtral as the
shipping ASR backend before the held-out corpus in issue #18 is complete.

## Evidence

| Dimension | Official MLX adapter | Pure-C/MPS control |
| --- | --- | --- |
| Runtime delivery | 149,789,936-byte current adapter, driver, and shared metallib | 246,024-byte executable |
| Model delivery | 3,133,798,126-byte 4-bit weights | 8,874,374,435-byte BF16 package |
| Measured GPU/runtime memory | 5.48 GB MLX peak on the 480 ms live run | 8.43 GB reported Metal weight cache; 16.66 GB process peak footprint |
| Streaming ownership | Hard queue, ingest/decode budgets, backpressure, cancellation, busy status | Synchronous `feed()` with no cancellation, busy, queue-capacity, or backpressure API |
| Functional control | Exact 177-token streaming-oracle text | One word substitution on the shared development fixture (`Halte` / `Haltet`) |
| Maintenance boundary | Official MLX release plus repository-owned model adapter | 45 commits across eleven days, no release tag, primary-author dominated, upstream warns that more production testing is needed |
| ADR-0005 disposition | Accepted foundation; model adapter owned here | Reference only |

The pure-C executable is dramatically smaller. That advantage does not offset
the larger model/working set, weaker lifecycle contract, or the maintenance
ownership Cuttledoc would acquire. The control remains valuable for readable
algorithm comparisons and future packaging hypotheses.

## Consequences

### Positive

- MLX remains a first-class platform foundation instead of being rejected
  because model integration requires owned C++ glue.
- Rust-facing lifecycle behavior is explicit, deterministic, and testable.
- Voxtral, Qwen ASR, future text generation, and possible TTS work can share
  one official Apple-local tensor runtime without sharing task contracts.
- The pure-C implementation remains available as a small, understandable
  control without entering the production dependency graph.

### Negative

- The current MLX runtime package is roughly 150 MB before pruning.
- Cuttledoc owns and must retest the Voxtral graph when either the model or MLX
  changes.
- The 4-bit model is still 3.13 GB and the measured live session peaks above
  5 GB of MLX allocation.
- Long audio, clean-host delivery, energy, timestamps, and held-out quality
  remain product gates outside this boundary decision.

## Validation

The decision is backed by the pinned
[`C/MPS control`](../../benchmarks/raw/phase0.voxtral-realtime-c-mps-control-1/result.json),
the [direct MLX streaming records](../../benchmarks/raw/phase0.voxtral-realtime-mlx-direct.streaming-480ms-320ms-1/result.json),
and their negative validator self-tests. Revisit this ADR only if a maintained
native boundary independently passes ADR-0005 and demonstrates the complete
Rust-owned lifecycle contract with materially better delivery or operation.
