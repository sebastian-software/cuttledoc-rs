# Apple-local ASR model evaluation

**Status:** mandatory legacy baselines, Apple Speech, and end-to-end official
MLX Whisper Tiny are measured on the first real quality fixture; broader
fixtures and comparable operating measurements remain before selection.

**Evidence date:** 2026-07-17.

**Machine-readable source:** [`benchmarks`](../benchmarks/).

## Current baseline state

| Candidate | Runtime boundary | WER / CER | Warm / RTF | Peak RSS | Model bytes | Timestamp / stream | State |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Parakeet TDT 0.6B v3 | Legacy Node addon + CoreML/VAD | 36.84% / 27.16% | 223.8 ms / 0.0212 | 145 MB | 1.90 GB downloaded | segment / final-only | measured |
| Whisper large-v3-turbo | Legacy Node addon + CoreML encoder + whisper.cpp | 21.05% / 3.70% | 657.6 ms / 0.0623 | 1.94 GB | 2.90 GB | segment / final-only | measured |
| Apple SpeechTranscriber | Repository-owned Swift C ABI | 5.26% / 1.23% | 198.2 ms / 0.0188 | 21.9 MB | system asset size unavailable | word / 27 volatile → 1 final | partial |
| Whisper Tiny | Repository-owned C++ task adapter over official MLX | 5.26% / 1.23% | 115.8 ms / 0.0110 | 124.7 MB process / 285.8 MB MLX peak | 74.4 MB model | segment / final-only | partial |

The MLX row is now a real end-to-end model path. The complete Whisper Tiny
frontend, encoder, autoregressive decoder, multilingual tokenizer, and
timestamp-token path ran from Rust without Python or Node. Metal FP16 and CPU
FP32 produced the same final text across repeated lifecycles. The text matches
the pinned official `mlx-whisper` 0.4.3 reference; the final segment boundary
differs by 100 ms because the C++ proof recomputes full decoder context instead
of using the reference KV cache. This advances MLX as a third first-class
foundation without treating one fixture as a production selection.

The numeric rows use one 10.56-second English FLEURS sample on an M1 Ultra with
64 GB RAM. Parakeet and legacy Whisper used five warm repetitions; Apple Speech
is one instrumented streamed run; MLX used three lifecycles with two runs each.
They are initial implementation baselines, not statistically meaningful model
rankings. The raw records retain timing values and exact hypotheses.

Whisper's recorded 1.07-second cold load is a new process with an existing
CoreML execution plan. The very first process after model download took
121.2 seconds to compile that plan. No equivalent clean-system number was
captured for Parakeet, so the cold columns must not be ranked against each
other yet.

## Artifact and license pins

| Artifact | Exact revision | Quantization / representation | License |
| --- | --- | --- | --- |
| FLEURS `en_us` test sample 0 | `cuttledoc@50b2d2d617fabce18a23f624fab4745ea1a792c4` import | Opus source normalized to f32le 16 kHz mono | CC-BY-4.0 |
| Parakeet CoreML | `FluidInference/parakeet-tdt-0.6b-v3-coreml@aed02740059203c4a87495924f685de3722ae9ce` | mixed Float16 and 6-bit palettized encoder | CC-BY-4.0 |
| Silero VAD CoreML | `FluidInference/silero-vad-coreml@b419383c55c110e2c9271fa6ee0ea83d03c70d96` | compiled CoreML v6.0.0 bundle | MIT |
| Whisper GGML | `ggerganov/whisper.cpp@5359861c739e955e79d9a303bcbc70fb988958b1` | upstream large-v3-turbo F16 GGML | MIT |
| Whisper CoreML encoder | `sebastian-software/whisper-coreml-models@dd3515371e6b560b63ec275abf020153a45caa60` | Float16 | MIT |
| Apple Speech asset | system-managed by macOS 26.5.2; revision not exposed | not exposed | Apple platform asset; not redistributed |
| Whisper Tiny MLX conversion | `mlx-community/whisper-tiny@78c52ab98ca87f570bc57ad852e15ef7060f9f76` | Float16 NPZ; all 166 model tensors used | MIT source model/converter; pinned model card omits structured license metadata |

The Parakeet model repository's structured metadata declares CC-BY-4.0 while
older prose associated with the conversion has also mentioned Apache-2.0.
The benchmark takes the more restrictive structured artifact declaration and
does not infer a different license from the original NVIDIA model.

## Harness decision

Schema 1.0.0 records exact host/runtime/model identity, per-artifact license,
quantization and options; WER/CER, cold/warm/RTF and first-result latency;
memory, artifact sizes and streaming update behavior; relative energy method;
engineering/update/packaging cost; raw evidence paths; and an explicit
measured/partial/blocked disposition. Candidate identifiers are deliberately
open so #12 reuses the schema unchanged.

The generated `say` fixture is smoke-only. It proves plumbing but cannot
influence model selection. The FLEURS source stays in the sibling repository;
the manifest pins its import commit, compressed-source digest, reference
digest, and the digest of the exact normalized PCM consumed by both candidates.
A measured result is rejected unless it uses a hashed, provenance-audited
quality fixture and supplies the mandatory comparison metrics.

The compatibility runner intentionally imports the legacy packages' CommonJS
entry points. The Parakeet ESM bundle currently fails on Node.js 24 because its
bundled `bindings` dependency performs a dynamic `require`; this packaging
defect is separate from ASR quality. Parakeet also emits token diagnostics to
stdout, so `--output` writes the machine-readable result independently.

## Next measurement order

1. Expand the quality set beyond one English clip and capture alternating
   energy plus comparable clean-system cold starts.
2. Use #12 selectively to determine whether Whisper Tiny is only boundary
   evidence or whether a stronger current MLX-native ASR model should enter the
   product bakeoff.
3. Only then choose the first vertical slice and fallback from measured product
   readiness, not runtime novelty.
