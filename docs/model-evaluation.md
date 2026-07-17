# Apple-local ASR model evaluation

**Status:** mandatory legacy baselines measured on the first real quality
fixture; Apple Speech and a real MLX ASR path remain before selection.

**Evidence date:** 2026-07-17.

**Machine-readable source:** [`benchmarks`](../benchmarks/).

## Current baseline state

| Candidate | Runtime boundary | WER / CER | Warm / RTF | Peak RSS | Model bytes | Timestamp / stream | State |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Parakeet TDT 0.6B v3 | Legacy Node addon + CoreML/VAD | 36.84% / 27.16% | 223.8 ms / 0.0212 | 145 MB | 1.90 GB downloaded | segment / final-only | measured |
| Whisper large-v3-turbo | Legacy Node addon + CoreML encoder + whisper.cpp | 21.05% / 3.70% | 657.6 ms / 0.0623 | 1.94 GB | 2.90 GB | segment / final-only | measured |
| Apple SpeechTranscriber | Repository-owned Swift C ABI | — | — | — | system asset | current shim has no update evidence | partial |
| MLX ASR candidate | Repository-owned C++ adapter over official MLX | — | — | — | — | model-dependent | precise blocker: no real ASR artifact has run |

The MLX row is a first-class candidate with a named model-path blocker. It is
not a rejection of the runtime: the direct official-core boundary and Metal
packaging path are already technically proven in #6.

The numeric rows use one 10.56-second English FLEURS sample and five warm
repetitions on an M1 Ultra with 64 GB RAM. They are initial implementation
baselines, not statistically meaningful model rankings. Whisper was more
accurate on this clip; Parakeet was about 2.9 times faster and used about
one-thirteenth of the peak RSS. The raw records retain every timing value and
the exact hypothesis.

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

1. Extend Apple Speech to expose timestamps, volatile/final updates, asset
   identity, cancellation, and raw timing/resource samples.
2. Select and run a real ASR artifact through the official MLX adapter, keeping
   the same fixture bytes and record schema.
3. Expand the quality set beyond one English clip and capture alternating
   energy plus comparable clean-system cold starts.
4. Only then choose the first vertical slice and fallback from measured product
   readiness, not runtime novelty.
