# Apple-local ASR model evaluation

**Status:** benchmark contract and bootstrap evidence implemented; no ASR
winner selected yet.

**Evidence date:** 2026-07-17.

**Machine-readable source:** [`benchmarks`](../benchmarks/).

## Current baseline state

| Candidate | Runtime boundary | Evidence | Current blocker |
| --- | --- | --- | --- |
| Parakeet TDT 0.6B v3 | Rust-owned CoreML worker | Blocked baseline record; real CoreML/VAD lifecycle proven separately | Decoder/model pipeline and licensed quality fixtures are not in this repository |
| Whisper large-v3-turbo | CoreML encoder + pinned whisper.cpp decoder | Blocked baseline record | Exact paired artifacts and compatibility runner are not in this repository |
| Apple SpeechTranscriber | Repository-owned Swift C ABI | Partial real file transcription | Current synthetic smoke run has no comparable timing, memory, energy, timestamps, confidence, or streaming data |
| MLX ASR candidate | Repository-owned C++ adapter over official MLX | Official MLX CPU/GPU model block runs from Rust | No real licensed ASR model path has been selected or executed |

The MLX row is a first-class candidate with a named model-path blocker. It is
not a rejection of the runtime: the direct official-core boundary and Metal
packaging path are already technically proven in #6.

## Harness decision

Schema 1.0.0 records exact host/runtime/model identity, per-artifact license,
quantization and options; WER/CER, cold/warm/RTF and first-result latency;
memory, artifact sizes and streaming update behavior; relative energy method;
engineering/update/packaging cost; raw evidence paths; and an explicit
measured/partial/blocked disposition. Candidate identifiers are deliberately
open so #12 reuses the schema unchanged.

The checked-in generated `say` fixture is smoke-only. It proves plumbing but
cannot influence model selection. A measured result is rejected unless it uses
a hashed, provenance-audited quality fixture and supplies the mandatory
comparison metrics. A non-redistributable fixture may be `local-required`; it
must never be silently checked in or treated as redistributable.

## Next measurement order

1. Import and provenance-audit the existing Parakeet and Whisper fixture set
   and exact model artifacts.
2. Add their long-lived benchmark adapters so cold load and warm inference are
   measured at the correct lifecycle boundary.
3. Extend Apple Speech to expose timestamps, volatile/final updates, asset
   identity, cancellation, and raw timing/resource samples.
4. Select and run a real ASR artifact through the official MLX adapter, keeping
   the same fixture bytes and record schema.
5. Only then choose the first vertical slice and fallback from measured product
   readiness, not runtime novelty.
