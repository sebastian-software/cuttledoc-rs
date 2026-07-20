# Direct Voxtral Realtime adapter over official MLX (#19)

This spike is the first repository-owned Voxtral Realtime boundary over the
official MLX C++ core. It deliberately replaces the `mlx-audio` session as the
product-boundary experiment without rejecting its model implementation as a
reference oracle.

The current milestone proves model identity, bounded incremental execution, and
exact fixed-fixture streaming parity through the complete model:

1. The C++ adapter loads the pinned 3.13 GB safetensors through official MLX
   and validates 1,523 tensors, their dtypes, 406 affine 4-bit modules, and
   critical encoder and decoder shapes.
2. Rust owns a narrow C ABI session with explicit queue, audio-ingest, and
   decode budgets, all-or-nothing feed backpressure, end-of-audio,
   cancellation, deterministic status codes, and append-only text deltas.
3. Each `step()` removes a fixed audio snapshot before executing official MLX
   work. New producer input therefore cannot extend the active step, which is
   the failure mode found in the reference runtime at 80 ms input cadence.
4. No MLX array, stream, model object, or C++ allocation crosses the ABI.
5. The same adapter reproduces the reference runtime's offline-streaming
   padding, 128-bin log-mel frontend, and two causal Conv1d/GELU stages using
   the pinned model weights and official MLX operations.
6. A reusable repository-owned rotating KV cache and sliding-window mask drive
   all 32 encoder transformer layers, RoPE, 4x downsampling, and the
   audio-language adapter without linking Python or `mlx-lm`.
7. Repository-owned streaming state retains the raw/mel frontier, two causal
   convolution buffers, 32 encoder caches, downsampling remainder, 26 decoder
   caches, token progress, and text progress across calls. Delay conditioning,
   grouped-query attention, the tied language-model head, greedy generation,
   and Tekken decoding produce append-only transcript updates through Rust.

These model procedures are implemented locally. Official MLX provides arrays,
operators, lazy evaluation, CPU kernels, and Metal kernels; it does not provide
or receive a call to a prebuilt Voxtral, sliding-window, or streaming-session
implementation.

The cache, causal-mask, sliding-window, quantized-linear, and activation
primitives are implemented in the shared `mlx-direct` support layer and are
model-independent. Voxtral still owns its layer graph, weight names,
dimensions, delay conditioning, generation policy, and tokenizer. Official MLX
provides tensor execution and Metal kernels; there is no call to a prebuilt
Voxtral implementation in MLX.

## Pinned inputs

- official MLX `v0.32.0`,
  `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`;
- official `mistralai/Voxtral-Mini-4B-Realtime-2602`,
  `2769294da9567371363522aac9bbcfdd19447add`;
- `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`,
  `fdebf7b2af834a1db4b8a3c99ab7480b333adf9e`.

The model and conversion are Apache-2.0. Official MLX is MIT. `mlx-audio`
remains a pinned reference-only oracle and is not linked by this adapter.

## Reproduce the boundary test

The default paths match the local Phase 0 workspace:

```sh
scripts/run-voxtral-mlx-direct-spike.sh
```

Override them when necessary:

```sh
CUTTLEDOC_MLX_SOURCE_DIR=/absolute/path/to/mlx-v0.32.0 \
CUTTLEDOC_VOXTRAL_MODEL_DIR=/absolute/path/to/model \
CUTTLEDOC_VOXTRAL_PCM_FIXTURE=/absolute/path/to/mono-16khz.f32le \
scripts/run-voxtral-mlx-direct-spike.sh
```

The checked German audiobook fixture contains 213,440 samples. The 480 ms
frontend probe applies the exact reference padding to 276,480 samples, produces
1,728 log-mel frames, and produces 864 vectors of width 1,280 after the causal
Conv1d stem. Against the pinned Python oracle, the maximum sampled absolute
error is `5.36e-7` and the maximum aggregate relative error is `3.64e-7`, below
the fixed `2e-6` and `2e-5` tolerances. A CPU smoke run preserves the output
shapes; the reproducible parity claim is based on the GPU run.

The same fixture drives the bounded lifecycle test. Filling the 10,240-sample
queue without stepping produces explicit backpressure on the next all-or-
nothing feed. Empty-input, closed-input, and cancelled states return their
pinned status codes; actual audio and decode work are exercised separately by
the live producer/consumer test.

The reference oracle and direct result are checked in at
`benchmarks/oracles/voxtral-realtime.audiobook-de-135_82_000105.frontend-480ms.json`
and
`benchmarks/raw/phase0.voxtral-realtime-mlx-direct.frontend-480ms-1/result.json`.
The runner validates them with `scripts/validate-voxtral-mlx-frontend.mjs`.

The encoder then processes the 864 stem frames as 750- and 114-frame chunks.
The second chunk uses an explicit `114 x 863` causal sliding-window mask. Its
cache has an absolute offset of 864 and a logical size of 750 while retaining
863 materialized frames during the multi-token update, matching the pinned
reference semantics. Across 14 encoder, adapter, and cache fingerprints, the
maximum sampled absolute error is `2.29e-5` (tolerance `5e-5`) and the maximum
aggregate relative error is `4.20e-6` (tolerance `1e-5`). The GPU probe takes
391 ms and peaks at 2.46 GB on the recorded M1 Ultra run. The checked-in oracle
and result use the corresponding `encoder-480ms` paths under
`benchmarks/oracles` and `benchmarks/raw`.

The batch decoder adds a 39-token delay-conditioned prompt and processes 216
adapter frames through 26 decoder layers. On the pinned fixture it emits the
same 178 token IDs and the exact same German transcript as the pinned
`mlx-audio` reference. Across decoder fingerprints, the maximum sampled
absolute error is `2.44e-4` (tolerance `5e-4`) and the maximum aggregate
relative error is `2.09e-4` (tolerance `5e-4`). The recorded M1 Ultra run takes
about 3.94 seconds for 13.34 seconds of audio and peaks at 5.07 GB before memory
optimization. The checked-in decoder oracle/result and
`scripts/validate-voxtral-mlx-decoder.mjs` make exact token and text parity a
hard gate.

## Incremental streaming result

The Rust driver creates the session before timing, captures PCM on an
independent wall-clock-paced producer thread, and repeatedly advances the
official-MLX executor until `DONE`. All three checked runs ingest each of the
213,440 samples exactly once, limit one step to 5,120 new samples and 16
decoded tokens, emit append-only deltas, and reproduce the pinned reference's
177-token streaming text exactly.

| Delay | Input cadence | First append | Maximum step | Endpoint finalization | MLX peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 480 ms | 320 ms | 1.71 s | 315 ms | 371 ms | 5.48 GB |
| 2,400 ms | 320 ms | 3.62 s | 337 ms | 727 ms | 5.54 GB |
| 480 ms | 80 ms | 1.64 s | 1.62 s | 791 ms | 5.48 GB |

The 320 ms cadence is the usable default from this development fixture. The
80 ms record is deliberately retained as a stress test: bounded snapshots
prevent the reference implementation's 25-second unbounded drain, but a
640 ms queue still applies heavy backpressure when an MLX step stalls. This is
correct bounded behavior, not evidence that 80 ms is production-ready.

The exact records live under
`benchmarks/raw/phase0.voxtral-realtime-mlx-direct.streaming-*`; the runner
checks text/token parity, append-only reconstruction, queue bounds, and a
five-second stress-regression ceiling against the pinned Python oracle.

Each checked run starts in a fresh process and measures 312-346 ms from session
construction through materialized encoder/decoder weights with the host file
cache warm. This is not a clean-host disk-download measurement. The recorded
local build consists of an 18,999,872-byte adapter dylib, a 625,912-byte Rust
driver, and the shared 130,164,152-byte MLX metallib, beside the separately
delivered 3,133,798,126-byte model. The manifest pins every model artifact;
clean-host delivery and cache-miss load remain product gates.

### Language control

The stable 320 ms path also completed ten fresh-process runs: one pinned
development audiobook each for German, English, Spanish, French, and
Portuguese at both delays. First-stable equals first-append because the direct
session never revokes emitted text.

| Language | 480 ms WER / CER | 2,400 ms WER / CER | First stable 480 / 2,400 |
| --- | ---: | ---: | ---: |
| German | 0.00% / 0.00% | 0.00% / 0.00% | 1.70 / 3.62 s |
| English | 5.88% / 0.45% | 9.80% / 6.31% | 2.02 / 3.94 s |
| Spanish | 18.42% / 4.29% | 10.53% / 2.45% | 1.38 / 3.29 s |
| French | 12.50% / 2.74% | 2.08% / 0.46% | 1.70 / 3.94 s |
| Portuguese | 0.00% / 0.00% | 0.00% / 0.00% | 1.70 / 3.63 s |

The five-clip macro WER improves from 7.36% to 4.48% at 2,400 ms, but that
average hides opposite language effects. Spanish and French improve, German
and Portuguese are unchanged, and English surface metrics worsen because the
later output renders `fifty thousand` as `50,000` in addition to the
`labourers`/`laborers` spelling difference. This is precisely why raw text,
CER, and error class remain alongside WER. The control is too small and too
development-exposed for model selection.

The complete ten-cell record is
`benchmarks/raw/phase0.voxtral-realtime-mlx-direct.streaming-language-control-1/result.json`.

## Remaining product gates

1. Expand the five-cell language control to the full multilingual audiobook
   pilot and genuinely held-out German-first podcast/audiobook gates through
   this repository-owned streaming boundary.
2. Compare clean-load, packaging, cancellation granularity, and maintenance
   cost with the pure-C/MPS control; do not duplicate quality evaluation.
3. If Voxtral remains selected, place this adapter behind the common engine
   contract, add long-audio cache/memory tests, and make the 320 ms default an
   explicit product setting rather than a spike constant.
