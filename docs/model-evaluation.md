# Apple-local ASR model evaluation

**Status:** the mandatory phase-0 comparison is complete. Apple
SpeechTranscriber is the selected first vertical-slice backend, Whisper
large-v3-turbo is the opt-in quality and coverage fallback, and direct official
MLX remains a first-class research foundation.

**Evidence date:** 2026-07-20.

**Machine-readable sources:**
[`phase0.multilingual-fleurs-10-1.json`](../benchmarks/matrices/phase0.multilingual-fleurs-10-1.json)
and its linked raw artifacts.

## Multilingual breadth result

Ten FLEURS test fixtures were used: two each for English, German, Spanish,
French, and Portuguese. Each candidate received one discarded warm-up followed
by two measured repetitions per fixture on the same M1 Ultra host.

| Candidate | Macro WER / CER | Warm / RTF | Peak RSS | Model / runtime bytes | Timestamp / stream | Disposition |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Apple SpeechTranscriber | 7.58% / 3.35% | 189.5 ms / 0.0136 | 24.0 MB | system-managed / 0.95 MB shim | word / incremental, 52.1 ms first result | primary |
| Qwen3-ASR 0.6B MLX reference | 5.10% / 1.49% | 356.7 ms / 0.0243 | 1.25 GB | 1.01 GB / 402.6 MB reference environment | token output / 148.0 ms first token; timestamps unknown | direct-MLX research |
| Whisper large-v3-turbo | 5.61% / 1.25% | 774.2 ms / 0.0564 | 2.00 GB | 2.90 GB / 2.17 MB addon | segment / final-only | fallback |
| Parakeet TDT 0.6B v3 | 10.27% / 4.75% | 249.8 ms / 0.0176 | 153.6 MB | 1.90 GB materialized / 0.13 MB addon | segment / final-only | compatibility baseline |
| Whisper Tiny on MLX | 26.24% / 7.61% | 284.8 ms / 0.0184 | 130.6 MB process | 74.4 MB / 148.8 MB MLX runtime | segment / final-only | runtime research |

Whisper large-v3-turbo has the second-best aggregate WER and best aggregate CER
in this bounded sample, but it uses about 83 times Apple's peak process memory,
has about four times Apple's warm latency, materializes a 2.90 GB model, and
only returns a final batch result. It is therefore valuable as an explicit
quality/coverage fallback rather than the default.

Apple SpeechTranscriber is the strongest product foundation for the supported
macOS 26 locales. It is close to the best measured quality while providing the
lowest warm latency and process RSS, real volatile-to-final updates, word
timestamps, and no repository-distributed model. Phase 2 should put the Rust
domain contract in front of the existing repository-owned Swift C ABI; product
logic must not move into Swift.

Qwen3-ASR 0.6B is the strongest new-model signal: the reference run slightly
beats Whisper's macro WER at less than half its warm latency and model size.
The result advances Qwen as the third candidate through an owned adapter over
official MLX. It does not make the community Python port product-ready, and it
does not displace Apple's default because it still uses about 52 times Apple's
process RSS, has slower first output, and has no aligned timestamps.

Parakeet remains compatibility evidence rather than a selected backend. In this
matrix it is less accurate than Apple SpeechTranscriber without adding
streaming, coverage, or a smaller materialized model.

The direct MLX result is not a reason to reject MLX. The complete Whisper Tiny
frontend, encoder, autoregressive decoder, multilingual tokenizer, and
timestamp-token path runs from Rust through a narrow repository-owned C++
adapter over the official MLX core, without Python or Node. Whisper Tiny's
multilingual quality is the limitation. Issue #12 should therefore test a
stronger current model that can use the same official MLX foundation rather
than replacing that foundation with an unmaintained wrapper.

## Language result

WER varies materially by language and candidate even in this small sample:

| Candidate | English | German | Spanish | French | Portuguese |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apple SpeechTranscriber | 7.39% | 7.54% | 6.15% | 1.75% | 15.04% |
| Qwen3-ASR 0.6B MLX reference | 7.39% | 4.76% | 7.61% | 0.00% | 5.74% |
| Whisper large-v3-turbo | 15.29% | 4.76% | 2.70% | 1.75% | 3.56% |
| Parakeet TDT 0.6B v3 | 23.18% | 13.10% | 2.70% | 5.26% | 7.13% |
| Whisper Tiny on MLX | 12.16% | 41.27% | 19.41% | 24.56% | 33.82% |

These values are regression and integration signals, not population-level model
rankings. Two samples per language are enough to expose language-token,
locale-resolution, and stream-reduction failures, but not enough to estimate
general recognition quality.

They nevertheless reject one global quality ranking. The provisional WER
leaders are Apple and Qwen for English, Qwen and Whisper for German, Whisper
and Parakeet for Spanish, Qwen for French, and Whisper for Portuguese. A
production selector may therefore accept language and operating mode as
inputs, but it must not auto-route from these two-sample estimates.

WER and CER here lowercase text and remove punctuation and whitespace before
comparison. A 5% WER is consequently not a 5% mixture of punctuation,
capitalization, and spelling mistakes: it represents word substitutions,
insertions, or omissions after normalization. Later evaluation must retain
this content-oriented metric while separately measuring surface form and
classifying errors by semantic severity.

## Artifact and license pins

| Artifact | Exact revision | Representation | License |
| --- | --- | --- | --- |
| FLEURS fixtures | `cuttledoc@50b2d2d617fabce18a23f624fab4745ea1a792c4` import | Opus sources normalized to f32le 16 kHz mono | CC-BY-4.0 |
| Parakeet CoreML | `FluidInference/parakeet-tdt-0.6b-v3-coreml@aed02740059203c4a87495924f685de3722ae9ce` | mixed Float16 and 6-bit palettized encoder | CC-BY-4.0 |
| Silero VAD CoreML | `FluidInference/silero-vad-coreml@b419383c55c110e2c9271fa6ee0ea83d03c70d96` | compiled CoreML v6.0.0 bundle | MIT |
| Whisper GGML | `ggerganov/whisper.cpp@5359861c739e955e79d9a303bcbc70fb988958b1` | large-v3-turbo F16 GGML | MIT |
| Whisper CoreML encoder | `sebastian-software/whisper-coreml-models@dd3515371e6b560b63ec275abf020153a45caa60` | Float16 | MIT |
| Apple Speech asset | system-managed by macOS 26.5.2; revision not exposed | not exposed | Apple platform asset; not redistributed |
| Whisper Tiny MLX conversion | `mlx-community/whisper-tiny@78c52ab98ca87f570bc57ad852e15ef7060f9f76` | Float16 NPZ | MIT source model/converter; converted model card has no structured license field |
| Qwen3-ASR 0.6B official / MLX reference | `Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0` / `mlx-community/Qwen3-ASR-0.6B-8bit@89e96d92ba34aca20b3e29fb10cc284097d1219f` | BF16 official / 8-bit MLX conversion | Apache-2.0 |
| Official MLX | `ml-explore/mlx@7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247` | v0.32.0 source build | MIT |

The Parakeet model repository's structured metadata declares CC-BY-4.0 while
older conversion prose has also mentioned Apache-2.0. The comparison uses the
more restrictive artifact declaration.

## Method and remaining uncertainty

Schema 1.0.0 records exact fixture, host, source, runtime, model, license and
boundary identity; WER/CER; cold, warm, RTF and first-result timing; process
memory, model/runtime size and streaming behavior; raw transcripts and
repetitions; and a measured disposition. The matrix validator cross-checks
every summary value against its immutable raw aggregate.

The generated `say` fixture remains smoke-only. Every quality fixture pins the
compressed source, reference, and normalized PCM digest. Apple locales are
explicitly mapped (`en-US`, `de-DE`, `es-MX`, `fr-FR`, and `pt-BR`) so
framework locale-equivalence choices cannot silently change a run. Apple
multi-range final updates are reduced with the same replace/revoke semantics as
the Rust stream contract.

All ten current quality fixtures come from FLEURS. That single-source,
short-utterance set is intentionally a breadth gate, not a proxy for
professionally recorded long-form content. The next quality set must report
language and source domain independently and add licensed audiobook plus
podcast material before any language-specific backend policy is accepted.

Energy is still unmeasured because the documented alternating `powermetrics`
procedure requires privileged sampling. Cold-load measurements used existing
system and model caches, and candidate order was not alternated. Those gaps
matter for later optimization and release thresholds, but they do not reverse
the first-backend decision across quality, latency, memory, package size,
timestamp detail, and streaming behavior.

## Next measurement order

1. Port the measured Qwen3-ASR 0.6B architecture through a bounded
   repository-owned C++ adapter over official MLX; keep the community runtime
   as reference-only.
2. Close the remaining CoreML acceptance gaps in #5 and resolve #3 against the
   selected Apple-primary and Whisper-fallback architecture.
3. Record the backend selection in an ADR, then begin the smallest Phase 1 /
   Phase 2 vertical slice justified by this evidence.
4. Treat alternating energy and clean-host cold starts as release-threshold
   follow-ups rather than reopening the runtime selection.
