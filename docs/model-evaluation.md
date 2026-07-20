# Apple-local ASR model evaluation

**Status:** the mandatory phase-0 comparison is complete. Apple
SpeechTranscriber is the selected first vertical-slice backend, Whisper
large-v3-turbo is the opt-in quality and coverage fallback, and direct official
MLX remains a first-class research foundation. Voxtral Realtime is now a
measured third model candidate with language-specific delay behavior.

**Evidence date:** 2026-07-20.

**Machine-readable sources:**
[`phase0.multilingual-fleurs-10-1.json`](../benchmarks/matrices/phase0.multilingual-fleurs-10-1.json)
and
[`phase0.audiobook-pilot-1.json`](../benchmarks/matrices/phase0.audiobook-pilot-1.json),
plus their linked raw artifacts.

## Multilingual breadth result

Ten FLEURS test fixtures were used: two each for English, German, Spanish,
French, and Portuguese. Each candidate received one discarded warm-up followed
by two measured repetitions per fixture on the same M1 Ultra host.

| Candidate | Macro WER / CER | Warm / RTF | Peak RSS | Model / runtime bytes | Timestamp / stream | Disposition |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Apple SpeechTranscriber | 7.58% / 3.35% | 189.5 ms / 0.0136 | 24.0 MB | system-managed / 0.95 MB shim | word / incremental, 52.1 ms first result | primary |
| Qwen3-ASR 0.6B MLX reference | 5.10% / 1.49% | 356.7 ms / 0.0243 | 1.25 GB | 1.01 GB / 402.6 MB reference environment | token output / 148.0 ms first token; timestamps unknown | direct-MLX research |
| Voxtral Realtime 4B MLX reference, 480 ms | 5.13% / 1.09% | 3,266.0 ms / 0.2233 | 3.41 GB | 3.15 GB / 475.4 MB reference environment | whole-clip result; live input not measured | research |
| Voxtral Realtime 4B MLX reference, 2,400 ms | 3.82% / 1.01% | 3,112.0 ms / 0.2140 | 3.40 GB | 3.15 GB / 475.4 MB reference environment | whole-clip result; live input not measured | research |
| Whisper large-v3-turbo | 5.61% / 1.25% | 774.2 ms / 0.0564 | 2.00 GB | 2.90 GB / 2.17 MB addon | segment / final-only | fallback |
| Parakeet TDT 0.6B v3 | 10.27% / 4.75% | 249.8 ms / 0.0176 | 153.6 MB | 1.90 GB materialized / 0.13 MB addon | segment / final-only | compatibility baseline |
| Whisper Tiny on MLX | 26.24% / 7.61% | 284.8 ms / 0.0184 | 130.6 MB process | 74.4 MB / 148.8 MB MLX runtime | segment / final-only | runtime research |

Voxtral at 2,400 ms has the best raw WER and CER in this bounded sample.
Whisper remains valuable as an explicit quality/coverage fallback because it
has an already accepted boundary and is about four times faster than Voxtral
offline. It still uses about 83 times Apple's peak process memory, has about
four times Apple's warm latency, materializes a 2.90 GB model, and only returns
a final batch result, so it is not the default.

Apple SpeechTranscriber is the strongest product foundation for the supported
macOS 26 locales. It is close to the best measured quality while providing the
lowest warm latency and process RSS, real volatile-to-final updates, word
timestamps, and no repository-distributed model. Phase 2 should put the Rust
domain contract in front of the existing repository-owned Swift C ABI; product
logic must not move into Swift.

Qwen3-ASR 0.6B remains the compact new-model signal: its recorded phase-0 WER
slightly beats Whisper at less than half Whisper's warm latency and model size.
Voxtral is the strongest raw-quality signal, including after the
boundary-preserving review, but uses substantially more time and memory. These
results advance both models as candidates through owned adapters over official
MLX. They do not make either community Python runtime product-ready, and they
do not displace Apple's default before held-out target-domain and true-streaming
evidence exists.

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
| Voxtral Realtime, 480 ms | 5.01% | 10.32% | 3.68% | 0.88% | 5.74% |
| Voxtral Realtime, 2,400 ms | 5.01% | 4.76% | 2.70% | 0.88% | 5.74% |
| Whisper large-v3-turbo | 15.29% | 4.76% | 2.70% | 1.75% | 3.56% |
| Parakeet TDT 0.6B v3 | 23.18% | 13.10% | 2.70% | 5.26% | 7.13% |
| Whisper Tiny on MLX | 12.16% | 41.27% | 19.41% | 24.56% | 33.82% |

These values are regression and integration signals, not population-level model
rankings. Two samples per language are enough to expose language-token,
locale-resolution, and stream-reduction failures, but not enough to estimate
general recognition quality.

They nevertheless reject one global quality ranking. Voxtral leads English
and is among the leaders in German, Spanish, and French; Qwen is exact on these
two French references, while Whisper leads Portuguese. A production selector
may therefore accept language and operating mode as inputs, but it must not
auto-route from these two-sample estimates.

WER and CER here lowercase text and remove punctuation and whitespace before
comparison. A 5% WER is consequently not a 5% mixture of punctuation,
capitalization, and spelling mistakes: it represents word substitutions,
insertions, or omissions after normalization. Later evaluation must retain
this content-oriented metric while separately measuring surface form and
classifying errors by semantic severity.

The deterministic
[`phase0.multilingual-fleurs-10-1.errors.json`](../benchmarks/analysis/phase0.multilingual-fleurs-10-1.errors.json)
trace reproduces the recorded WER and adds a boundary-preserving review
alignment grouped by candidate and language. The second view no longer counts
`T-Rex` as `trex` or `25-30` as `2530`; it does not silently equate semantically
different numbers. Its risk hints are mechanical, and semantic severity remains
deliberately unreviewed until a human classifies the changed content.

The review view changes the aggregate and language numbers as follows:

| Candidate | Aggregate | English | German | Spanish | French | Portuguese |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple SpeechTranscriber | 6.62% | 7.39% | 2.78% | 6.15% | 1.72% | 15.04% |
| Qwen3-ASR 0.6B MLX reference | 4.15% | 7.39% | 0.00% | 7.61% | 0.00% | 5.74% |
| Voxtral Realtime, 480 ms | 4.17% | 5.01% | 5.56% | 3.68% | 0.86% | 5.74% |
| Voxtral Realtime, 2,400 ms | 2.86% | 5.01% | 0.00% | 2.70% | 0.86% | 5.74% |
| Whisper large-v3-turbo | 3.60% | 10.03% | 0.00% | 2.70% | 1.72% | 3.56% |
| Parakeet TDT 0.6B v3 | 9.31% | 23.18% | 8.33% | 2.70% | 5.22% | 7.13% |
| Whisper Tiny on MLX | 25.41% | 12.16% | 36.51% | 19.41% | 25.18% | 33.82% |

This still is not semantic WER. For example, `3` versus `III` remains a
reviewable numeric representation difference, while `3` versus `XI` is a
critical content error. The report flags both rather than silently deciding
that they are equivalent.

## Voxtral Realtime cross-source follow-up

The pinned 4-bit Voxtral Realtime MLX oracle completed the 15 professionally
recorded audiobook clips at both the publisher-recommended 480 ms delay and a
2,400 ms quality control:

| Delay | Macro WER | German | English | Spanish | French | Portuguese | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 480 ms | 5.84% | 2.72% | 5.41% | 7.94% | 6.68% | 6.45% | 0.223 |
| 2,400 ms | 4.25% | 0.68% | 6.06% | 5.31% | 1.74% | 7.46% | 0.225 |
| Whisper | 4.49% | 2.52% | 5.95% | 1.21% | 4.77% | 8.00% | 0.0569 |

Voxtral at 2,400 ms has the lowest raw macro WER in this development pilot,
while 480 ms is better for English and Portuguese. The artifact is
Apache-2.0 and Apple-local feasibility is proven, but it is substantially
heavier: a 3.15 GB model, roughly 5.8 GB MLX peak allocation, and about four
times Whisper's offline RTF. The runner loads complete clips, so no live-input
latency or cancellation claim follows from these numbers.

The second-source FLEURS control preserves the 2,400 ms aggregate advantage
(3.82% versus 5.13%) and the strong French result. It does not preserve every
per-language direction: German improves from 10.32% to 4.76%, while English,
French, and Portuguese are identical across delays. German's remaining raw WER
at 2,400 ms drops to 0% in the boundary-review view because the differences are
word boundaries such as `Kow Loon` versus `Kowloon`, not character changes.
This is useful evidence that Voxtral deserves intensive evaluation, while also
showing that delay tuning must be judged by source, language, CER, and error
class rather than a global WER alone.

## Voxtral Realtime live-input follow-up

A separate probe fed one 13.34-second German audiobook clip from a producer
thread at wall-clock pace while the MLX executor called the stateful session
cooperatively. Both delays, two repeated lifecycles, and two input chunk sizes
produced the same append-only transcript at 0% WER/CER:

| Input chunk | Delay | First append | Maximum running `step()` | Endpoint finalization |
| --- | ---: | ---: | ---: | ---: |
| 80 ms | 480 ms | 1.63-1.66 s | 6.24-13.75 s | 5.24-5.38 s |
| 80 ms | 2,400 ms | 3.55-28.12 s | 0.13-25.15 s | 3.30-16.76 s |
| 320 ms | 480 ms | 1.69-1.71 s | 0.13-0.20 s | 0.36-0.50 s |
| 320 ms | 2,400 ms | 3.61 s | 0.13-0.14 s | 0.66-0.72 s |

The configured transcription delay is therefore not total startup latency:
left context, prefill, chunk cadence, and executor time also contribute. This
single development-exposed, dataset-transcript-unverified fixture is an
operational control, not a German quality estimate.

The 320 ms control proves usable incremental model behavior. The 80 ms trace
also exposes a reference-runtime defect: `mlx-audio` drains its pending audio
list in a loop while the independent producer can append more work. Once the
consumer falls behind, one documented bounded `step()` can keep ingesting until
end-of-audio. Producer p95 scheduling lateness remains about 5 ms, so the
multi-second stalls are not microphone scheduling noise.

The exact
[`80 ms trace`](../benchmarks/raw/phase0.voxtral-realtime-mlx-reference.streaming-80ms-1/result.json)
and
[`320 ms trace`](../benchmarks/raw/phase0.voxtral-realtime-mlx-reference.streaming-320ms-1/result.json)
retain all 51 append deltas and every executor step. The runtime exposes no
`cancel()` method, and `close()` performs end-of-stream padding and
finalization rather than cancellation. Its model docstring also names a
`finalize_step()` method that the session does not implement.

This rejects `mlx-audio` as Cuttledoc's product boundary, not official MLX or
Voxtral. The candidate advances to a bounded repository-owned ingestion loop
over official MLX, with cooperative cancellation designed into the Rust-owned
lifecycle. The pure-C/MPS implementation remains a comparison boundary, not a
reason to abandon the first-class MLX platform.

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
| Voxtral Realtime 4B official / MLX reference | `mistralai/Voxtral-Mini-4B-Realtime-2602@2769294da9567371363522aac9bbcfdd19447add` / `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit@fdebf7b2af834a1db4b8a3c99ab7480b333adf9e` | BF16 official / 4-bit MLX conversion | Apache-2.0 |
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

The original ten-fixture quality breadth gate comes from FLEURS. The additional
15-fixture MLS/LibriSpeech development pilot now adds professionally recorded
LibriVox audiobook audio across the same five languages, three
speakers/chapters each. It is still too small and already inspected, so
held-out audiobook plus licensed professional-podcast material remains required
before any language-specific backend policy is accepted.

Energy is still unmeasured because the documented alternating `powermetrics`
procedure requires privileged sampling. Cold-load measurements used existing
system and model caches, and candidate order was not alternated. Those gaps
matter for later optimization and release thresholds, but they do not reverse
the first-backend decision across quality, latency, memory, package size,
timestamp detail, and streaming behavior.

## Next measurement order

1. Treat the end-to-end Qwen3-ASR 0.6B spike under
   [#17](https://github.com/sebastian-software/cuttledoc-rs/issues/17) as
   complete. The direct adapter completed the 15-fixture audiobook pilot with
   10.96% macro WER, matched the reference text on 12/15 clips, and passed
   repeated Rust-owned lifecycle, exact transcript, busy, error, and
   cancellation checks. Keep the community runtime as reference-only. See the
   [direct Qwen3-ASR spike](spikes/qwen3-mlx-direct.md).
2. Treat Voxtral Realtime as a measured model candidate. Preserve the proven
   320 ms live-input behavior in a repository-owned official-MLX adapter with
   bounded ingestion and cooperative cancellation; compare that narrow
   boundary with pure C/MPS without adopting `mlx-audio`.
3. Acquire held-out German-first professional-podcast material and independent
   audiobook works, then rerun Apple, Whisper, direct Qwen, Parakeet, and
   Voxtral on the identical language/domain cells.
4. Close the remaining CoreML acceptance gaps in #5 and resolve #3 against the
   selected Apple-primary and Whisper-fallback architecture.
5. Record the backend selection in an ADR, then begin the smallest Phase 1 /
   Phase 2 vertical slice justified by this evidence.
6. Evaluate the frozen postprocessing prompt/model tuples on the held-out raw
   outputs; never fold corrected text into the raw backend ranking.
7. Treat alternating energy and clean-host cold starts as release-threshold
   follow-ups rather than reopening the runtime selection.
